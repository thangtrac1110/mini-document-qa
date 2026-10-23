"""Ingestion and search orchestration over in-memory state.

State decisions:
- All state lives in memory and is lost when the process restarts.
- A document is visible (listed and searchable) only after ingestion fully succeeds. The
  new index is built from the existing chunks plus the new ones *before* anything is
  committed; if any step fails, the previous documents and index stay untouched.
- Re-uploading byte-identical content raises ``DuplicateDocumentError`` (detected by
  SHA-256). The same filename with different content is a separate document.
- Search covers all documents, or one document via ``document_id``. IDF statistics are
  always computed over the whole corpus, so a filtered search ranks a document's chunks
  with corpus-wide term weights.
"""

import hashlib
import threading
import uuid
from dataclasses import dataclass

from app.config import Settings
from app.errors import DocumentNotFoundError, DuplicateDocumentError
from app.ingestion.chunking import Chunk, chunk_pages
from app.ingestion.pdf_reader import extract_pages
from app.retrieval.bm25_index import BM25Index


@dataclass(frozen=True)
class Document:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    sha256: str


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    score: float


class DocumentService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._documents: dict[str, Document] = {}
        self._chunks: list[Chunk] = []
        self._index = BM25Index([])
        self._lock = threading.Lock()

    def ingest_pdf(self, filename: str, data: bytes) -> Document:
        sha256 = hashlib.sha256(data).hexdigest()
        existing = self._find_by_hash(sha256)
        if existing:
            raise DuplicateDocumentError(existing.document_id)

        # Extraction and chunking are pure; they need no lock and leave no state on failure.
        document_id = uuid.uuid4().hex
        pages = extract_pages(data)
        chunks = chunk_pages(document_id, pages, self.settings.chunk_size_words)
        document = Document(document_id, filename, len(pages), len(chunks), sha256)

        with self._lock:
            # Re-check under the lock in case the same file was ingested concurrently.
            existing = self._find_by_hash(sha256)
            if existing:
                raise DuplicateDocumentError(existing.document_id)
            all_chunks = self._chunks + chunks
            new_index = self._build_index(all_chunks)
            # Commit only after every step succeeded.
            self._chunks = all_chunks
            self._index = new_index
            self._documents[document_id] = document
        return document

    def list_documents(self) -> list[Document]:
        return list(self._documents.values())

    def get_document(self, document_id: str) -> Document:
        try:
            return self._documents[document_id]
        except KeyError:
            raise DocumentNotFoundError(document_id) from None

    def get_chunks(self, document_id: str) -> list[Chunk]:
        self.get_document(document_id)
        return [c for c in self._chunks if c.document_id == document_id]

    def search(self, query: str, top_k: int, document_id: str | None = None) -> list[SearchHit]:
        if document_id is not None:
            self.get_document(document_id)
        index, documents = self._index, self._documents  # consistent snapshot
        return [
            SearchHit(
                chunk_id=r.chunk.chunk_id,
                document_id=r.chunk.document_id,
                filename=documents[r.chunk.document_id].filename,
                page_number=r.chunk.page_number,
                text=r.chunk.text,
                score=r.score,
            )
            for r in index.search(query, top_k, document_id)
        ]

    def _find_by_hash(self, sha256: str) -> Document | None:
        return next((d for d in self._documents.values() if d.sha256 == sha256), None)

    def _build_index(self, chunks: list[Chunk]) -> BM25Index:
        # Separate method so tests can simulate an indexing failure.
        return BM25Index(chunks)
