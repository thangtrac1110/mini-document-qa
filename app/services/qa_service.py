"""Question answering: BM25 retrieval -> LLM generation -> citation validation.

Behavior:
- Retrieval reuses ``DocumentService.search`` unchanged, so /search and /ask rank alike.
- If retrieval finds nothing, the LLM is not called and the result says the documents
  do not contain an answer (``insufficient_information=True``, ``llm_called=False``).
- Only the retrieved top-k chunks and the question are sent to the model.
- Source IDs returned by the model are checked against the IDs actually sent. Unknown
  IDs are dropped from the citations and reported in ``rejected_source_ids``.
- ``grounded`` is computed by the backend: the model claimed enough information *and*
  cited at least one real source. It means "has valid citations", not "verified true".
"""

from dataclasses import dataclass

from app.errors import LLMNotConfiguredError
from app.generation.answer_generator import AnswerGenerator
from app.generation.prompts import ContextSource, label_sources
from app.pipeline import DocumentService

NO_EVIDENCE_ANSWER = (
    "No passages in the uploaded documents match this question, so it cannot be answered from them."
)


@dataclass(frozen=True)
class AskResult:
    question: str
    answer: str
    insufficient_information: bool
    grounded: bool
    llm_called: bool
    sources: list[ContextSource]  # everything the model was shown
    cited_source_ids: list[str]  # validated subset of sources, in the model's order
    rejected_source_ids: list[str]


class QAService:
    def __init__(self, documents: DocumentService, generator: AnswerGenerator | None):
        self._documents = documents
        self._generator = generator

    @property
    def configured(self) -> bool:
        return self._generator is not None

    def ask(self, question: str, top_k: int, document_id: str | None = None) -> AskResult:
        if self._generator is None:
            raise LLMNotConfiguredError("Answer generation is not configured: set ANTHROPIC_API_KEY")

        hits = self._documents.search(question, top_k, document_id)
        if not hits:
            return AskResult(question, NO_EVIDENCE_ANSWER, True, False, False, [], [], [])

        sources = label_sources(hits)
        generated = self._generator.generate(question, sources)

        known = {s.source_id for s in sources}
        cited, rejected = [], []
        for source_id in dict.fromkeys(generated.supporting_source_ids):  # dedupe, keep order
            (cited if source_id in known else rejected).append(source_id)

        return AskResult(
            question=question,
            answer=generated.answer,
            insufficient_information=generated.insufficient_information,
            grounded=not generated.insufficient_information and bool(cited),
            llm_called=True,
            sources=sources,
            cited_source_ids=cited,
            rejected_source_ids=rejected,
        )
