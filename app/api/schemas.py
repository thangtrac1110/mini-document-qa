"""Request and response models for the HTTP API."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

MAX_TOP_K = 50
MAX_ASK_TOP_K = 20  # bounds how much document text is sent to the LLM per question

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class SearchRequest(BaseModel):
    query: QueryText
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    document_id: str | None = Field(default=None, description="Restrict search to one document")


class SearchResultOut(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(description="1-based page number in the source PDF")
    text: str
    score: float = Field(description="BM25 score; a relative ranking value, not a probability")


class SearchResponse(BaseModel):
    query: str
    top_k: int
    document_id: str | None
    results: list[SearchResultOut]


class AskRequest(BaseModel):
    question: QueryText
    top_k: int = Field(default=5, ge=1, le=MAX_ASK_TOP_K, description="Passages sent to the LLM")
    document_id: str | None = Field(default=None, description="Restrict retrieval to one document")


class SourceOut(SearchResultOut):
    source_id: str = Field(description='Label shown to the LLM, e.g. "S1"')


class CitationOut(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    filename: str
    page_number: int


class AskResponse(BaseModel):
    question: str
    answer: str
    insufficient_information: bool = Field(description="The model (or retrieval) found too little evidence")
    grounded: bool = Field(
        description="Answer claims sufficient evidence and cites at least one retrieved source. "
        "Computed by the backend; it does not prove the answer is correct."
    )
    llm_called: bool
    citations: list[CitationOut] = Field(description="Validated citations; metadata comes from the backend")
    rejected_source_ids: list[str] = Field(description="IDs the model returned that were not in its context")
    sources: list[SourceOut] = Field(description="All retrieved passages that were sent to the model")


class ErrorResponse(BaseModel):
    detail: str
