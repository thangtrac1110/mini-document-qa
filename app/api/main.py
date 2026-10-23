"""FastAPI application. Route handlers only validate HTTP input and translate results;
ingestion and ranking live in ``app.pipeline.DocumentService`` and question answering in
``app.services.qa_service.QAService``.

Run with: uvicorn app.api.main:app --env-file .env
"""

import os
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    DocumentOut,
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    SearchResultOut,
    SourceOut,
)
from app.config import Settings
from app.errors import (
    AppError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmptyQueryError,
    InvalidParameterError,
    InvalidPDFError,
    LLMConfigurationError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    MalformedLLMResponseError,
    NoExtractableTextError,
)
from app.generation.answer_generator import AnswerGenerator
from app.generation.llm_client import AnthropicLLMClient, LLMClient
from app.pipeline import Document, DocumentService
from app.services.qa_service import AskResult, QAService

ERROR_STATUS: dict[type[AppError], int] = {
    InvalidPDFError: 400,
    DocumentNotFoundError: 404,
    DuplicateDocumentError: 409,
    NoExtractableTextError: 422,
    EmptyQueryError: 422,
    InvalidParameterError: 422,
    LLMProviderError: 502,
    LLMRefusalError: 502,
    MalformedLLMResponseError: 502,
    LLMNotConfiguredError: 503,
    LLMConfigurationError: 503,
    LLMRateLimitError: 503,
    LLMTimeoutError: 504,
}

_FROM_ENV = object()  # sentinel: build the LLM client from ANTHROPIC_API_KEY


def get_service(request: Request) -> DocumentService:
    return request.app.state.service


def get_qa_service(request: Request) -> QAService:
    return request.app.state.qa_service


Service = Annotated[DocumentService, Depends(get_service)]
QA = Annotated[QAService, Depends(get_qa_service)]


def to_document_out(doc: Document) -> DocumentOut:
    return DocumentOut(
        document_id=doc.document_id,
        filename=doc.filename,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
    )


def to_ask_response(result: AskResult) -> AskResponse:
    by_id = {s.source_id: s.hit for s in result.sources}
    return AskResponse(
        question=result.question,
        answer=result.answer,
        insufficient_information=result.insufficient_information,
        grounded=result.grounded,
        llm_called=result.llm_called,
        citations=[
            CitationOut(
                source_id=sid,
                chunk_id=by_id[sid].chunk_id,
                document_id=by_id[sid].document_id,
                filename=by_id[sid].filename,
                page_number=by_id[sid].page_number,
            )
            for sid in result.cited_source_ids
        ],
        rejected_source_ids=result.rejected_source_ids,
        sources=[SourceOut(source_id=s.source_id, **asdict(s.hit)) for s in result.sources],
    )


def build_llm_client(settings: Settings) -> LLMClient | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    return AnthropicLLMClient(api_key, settings) if api_key else None


def create_app(settings: Settings | None = None, llm_client: LLMClient | None | object = _FROM_ENV) -> FastAPI:
    """``llm_client`` defaults to a client built from the environment; pass ``None`` to
    disable answer generation or a fake client in tests."""
    settings = settings or Settings.from_env()
    if llm_client is _FROM_ENV:
        llm_client = build_llm_client(settings)

    app = FastAPI(
        title="Mini Document Q&A",
        description="Upload text-based PDFs, retrieve passages with BM25 and ask questions answered "
        "by an LLM from those passages. Documents are kept in memory and are lost on restart.",
        version="0.2.0",
    )
    app.state.service = DocumentService(settings)
    app.state.qa_service = QAService(app.state.service, AnswerGenerator(llm_client) if llm_client else None)

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        status = next((code for cls, code in ERROR_STATUS.items() if isinstance(exc, cls)), 500)
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    errors = {code: {"model": ErrorResponse} for code in (400, 404, 409, 413, 415, 422)}
    ask_errors = {code: {"model": ErrorResponse} for code in (404, 422, 502, 503, 504)}

    @app.get("/health")
    def health(qa: QA) -> dict[str, str | bool]:
        return {"status": "ok", "llm_configured": qa.configured}

    @app.post("/documents", status_code=201, response_model=DocumentOut, responses=errors)
    async def upload_document(service: Service, file: UploadFile = File(...)) -> DocumentOut:
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(415, "Only .pdf files are accepted")

        limit = service.settings.max_upload_bytes
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"File exceeds the {limit // (1024 * 1024)} MB limit")
        if not data:
            raise HTTPException(400, "Uploaded file is empty")
        # The PDF spec allows the header anywhere in the first 1024 bytes.
        if b"%PDF-" not in data[:1024]:
            raise HTTPException(415, "File content is not a PDF")

        return to_document_out(service.ingest_pdf(filename, data))

    @app.get("/documents", response_model=list[DocumentOut])
    def list_documents(service: Service) -> list[DocumentOut]:
        return [to_document_out(d) for d in service.list_documents()]

    @app.get("/documents/{document_id}", response_model=DocumentOut, responses=errors)
    def get_document(document_id: str, service: Service) -> DocumentOut:
        return to_document_out(service.get_document(document_id))

    @app.post("/search", response_model=SearchResponse, responses=errors)
    def search(body: SearchRequest, service: Service) -> SearchResponse:
        hits = service.search(body.query, body.top_k, body.document_id)
        return SearchResponse(
            query=body.query,
            top_k=body.top_k,
            document_id=body.document_id,
            results=[SearchResultOut(**asdict(h)) for h in hits],
        )

    # Sync handler: FastAPI runs it in a worker thread, so the blocking LLM call does not
    # stall the event loop.
    @app.post("/ask", response_model=AskResponse, responses=ask_errors)
    def ask(body: AskRequest, qa: QA) -> AskResponse:
        return to_ask_response(qa.ask(body.question, body.top_k, body.document_id))

    return app


app = create_app()
