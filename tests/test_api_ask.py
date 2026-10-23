import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    MalformedLLMResponseError,
)


def make_client(llm, make_pdf) -> TestClient:
    client = TestClient(create_app(Settings(chunk_size_words=20), llm_client=llm))
    pdf = make_pdf(["Owls hunt at night using sharp hearing.", "Robins eat worms in spring."])
    assert client.post("/documents", files={"file": ("birds.pdf", pdf, "application/pdf")}).status_code == 201
    return client


def test_ask_returns_answer_with_backend_citations(fake_llm, make_pdf):
    llm = fake_llm({"answer": "They hunt at night.", "supporting_source_ids": ["S1", "S5"],
                    "insufficient_information": False})
    client = make_client(llm, make_pdf)

    response = client.post("/ask", json={"question": "How do owls hunt?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "They hunt at night."
    assert body["grounded"] is True
    assert body["llm_called"] is True
    assert body["rejected_source_ids"] == ["S5"]
    (citation,) = body["citations"]
    assert citation["source_id"] == "S1"
    assert citation["filename"] == "birds.pdf"
    assert citation["page_number"] == 1
    assert body["sources"][0]["source_id"] == "S1"
    assert "Owls hunt" in body["sources"][0]["text"]


def test_ask_without_matching_passages_does_not_call_llm(fake_llm, make_pdf):
    llm = fake_llm()
    client = make_client(llm, make_pdf)

    response = client.post("/ask", json={"question": "quantum entanglement"})

    assert response.status_code == 200
    assert response.json()["insufficient_information"] is True
    assert response.json()["llm_called"] is False
    assert llm.calls == []


def test_ask_when_llm_not_configured_returns_503_but_search_still_works(make_pdf):
    client = make_client(None, make_pdf)

    assert client.get("/health").json()["llm_configured"] is False
    assert client.post("/ask", json={"question": "owls"}).status_code == 503
    assert client.post("/search", json={"query": "owls"}).status_code == 200


def test_search_never_calls_the_llm(fake_llm, make_pdf):
    llm = fake_llm()
    client = make_client(llm, make_pdf)

    assert client.post("/search", json={"query": "owls"}).status_code == 200
    assert llm.calls == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LLMTimeoutError("The LLM provider timed out"), 504),
        (LLMRateLimitError("rate limited"), 503),
        (LLMConfigurationError("bad credentials"), 503),
        (LLMProviderError("HTTP 500"), 502),
        (LLMRefusalError("declined"), 502),
        (MalformedLLMResponseError("not json"), 502),
    ],
)
def test_llm_failures_map_to_controlled_errors(fake_llm, make_pdf, error, status):
    client = make_client(fake_llm(error), make_pdf)

    response = client.post("/ask", json={"question": "owls"})

    assert response.status_code == status
    assert response.json() == {"detail": str(error)}


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "  "}, {"question": "ok", "top_k": 0}, {"question": "ok", "top_k": 21}, {}],
)
def test_invalid_ask_requests_return_422(fake_llm, make_pdf, payload):
    assert make_client(fake_llm(), make_pdf).post("/ask", json=payload).status_code == 422


def test_ask_unknown_document_returns_404(fake_llm, make_pdf):
    response = make_client(fake_llm(), make_pdf).post("/ask", json={"question": "owls", "document_id": "nope"})

    assert response.status_code == 404
