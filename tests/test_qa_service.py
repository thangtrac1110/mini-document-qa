import pytest

from app.config import Settings
from app.errors import (
    DocumentNotFoundError,
    LLMNotConfiguredError,
    LLMTimeoutError,
    MalformedLLMResponseError,
)
from app.generation.answer_generator import AnswerGenerator
from app.pipeline import DocumentService
from app.services.qa_service import NO_EVIDENCE_ANSWER, QAService


@pytest.fixture
def documents(make_pdf):
    service = DocumentService(Settings(chunk_size_words=20))
    service.ingest_pdf("birds.pdf", make_pdf(["Owls hunt at night using sharp hearing.", "Robins eat worms."]))
    service.ingest_pdf("space.pdf", make_pdf(["Mars has two small moons."]))
    return service


def reply(answer="ok", ids=("S1",), insufficient=False) -> dict:
    return {"answer": answer, "supporting_source_ids": list(ids), "insufficient_information": insufficient}


def make_qa(documents, llm) -> QAService:
    return QAService(documents, AnswerGenerator(llm))


def test_answer_with_valid_citation(documents, fake_llm):
    llm = fake_llm(reply("Owls hunt at night.", ["S1"]))

    result = make_qa(documents, llm).ask("how do owls hunt", top_k=3)

    assert result.answer == "Owls hunt at night."
    assert result.llm_called and result.grounded
    assert result.cited_source_ids == ["S1"]
    assert result.rejected_source_ids == []
    assert result.sources[0].hit.filename == "birds.pdf"
    assert result.sources[0].hit.page_number == 1


def test_only_retrieved_context_and_question_are_sent(documents, fake_llm):
    llm = fake_llm()

    result = make_qa(documents, llm).ask("owls hunt", top_k=1)

    (call,) = llm.calls
    assert len(result.sources) == 1
    assert "Owls hunt at night" in call["user"]
    assert "owls hunt" in call["user"]
    assert "Mars" not in call["user"]
    assert "Robins" not in call["user"]


def test_invented_and_duplicate_source_ids_are_handled(documents, fake_llm):
    llm = fake_llm(reply(ids=["S1", "S7", "S1", "birds.pdf"]))

    result = make_qa(documents, llm).ask("owls hunt", top_k=2)

    assert result.cited_source_ids == ["S1"]
    assert result.rejected_source_ids == ["S7", "birds.pdf"]


def test_answer_without_valid_citation_is_not_grounded(documents, fake_llm):
    llm = fake_llm(reply("Owls are birds.", ids=["S9"]))

    result = make_qa(documents, llm).ask("owls hunt", top_k=2)

    assert not result.insufficient_information
    assert not result.grounded
    assert result.cited_source_ids == []


def test_model_abstention_is_passed_through(documents, fake_llm):
    llm = fake_llm(reply("The sources do not say.", ids=[], insufficient=True))

    result = make_qa(documents, llm).ask("what do owls weigh", top_k=3)

    assert result.insufficient_information
    assert not result.grounded
    assert result.llm_called


def test_no_retrieved_passages_skips_the_llm(documents, fake_llm):
    llm = fake_llm()

    result = make_qa(documents, llm).ask("quantum entanglement", top_k=3)

    assert llm.calls == []
    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.insufficient_information and not result.llm_called and not result.grounded
    assert result.sources == []


def test_document_filter_limits_context(documents, fake_llm):
    llm = fake_llm()
    space_id = next(d.document_id for d in documents.list_documents() if d.filename == "space.pdf")

    result = make_qa(documents, llm).ask("small moons owls", top_k=5, document_id=space_id)

    assert {s.hit.filename for s in result.sources} == {"space.pdf"}


def test_unknown_document_filter_raises(documents, fake_llm):
    with pytest.raises(DocumentNotFoundError):
        make_qa(documents, fake_llm()).ask("owls", top_k=3, document_id="missing")


def test_not_configured_raises(documents):
    qa = QAService(documents, None)

    assert not qa.configured
    with pytest.raises(LLMNotConfiguredError):
        qa.ask("owls", top_k=3)


def test_provider_errors_propagate(documents, fake_llm):
    with pytest.raises(LLMTimeoutError):
        make_qa(documents, fake_llm(LLMTimeoutError("timed out"))).ask("owls", top_k=3)


@pytest.mark.parametrize(
    "bad_reply",
    [
        {"answer": "x"},
        {"answer": 1, "supporting_source_ids": [], "insufficient_information": False},
        {**reply(), "extra": True},
    ],
)
def test_schema_violations_raise(documents, fake_llm, bad_reply):
    with pytest.raises(MalformedLLMResponseError):
        make_qa(documents, fake_llm(bad_reply)).ask("owls", top_k=3)
