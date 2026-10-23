import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings


@pytest.fixture
def client():
    return TestClient(create_app(Settings(chunk_size_words=20, max_upload_bytes=50_000), llm_client=None))


def upload(client, data: bytes, filename: str = "doc.pdf"):
    return client.post("/documents", files={"file": (filename, data, "application/pdf")})


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "llm_configured": False}


def test_upload_list_get_and_search(client, make_pdf):
    created = upload(client, make_pdf(["Bees pollinate flowers.", "Owls hunt at night using sharp hearing."]))
    assert created.status_code == 201
    doc = created.json()
    assert doc["filename"] == "doc.pdf"
    assert doc["page_count"] == 2
    assert doc["chunk_count"] == 2

    assert client.get("/documents").json() == [doc]
    assert client.get(f"/documents/{doc['document_id']}").json() == doc

    response = client.post("/search", json={"query": "how do owls hunt", "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "how do owls hunt"
    assert body["top_k"] == 1
    assert len(body["results"]) == 1
    hit = body["results"][0]
    assert hit["document_id"] == doc["document_id"]
    assert hit["filename"] == "doc.pdf"
    assert hit["page_number"] == 2
    assert "Owls" in hit["text"]
    assert hit["score"] > 0


def test_search_with_document_filter(client, make_pdf):
    a = upload(client, make_pdf(["river water flows downhill"]), "a.pdf").json()
    b = upload(client, make_pdf(["water boils at 100 degrees"]), "b.pdf").json()

    response = client.post("/search", json={"query": "water", "document_id": b["document_id"]})

    assert response.status_code == 200
    assert [r["document_id"] for r in response.json()["results"]] == [b["document_id"]]
    assert a["document_id"] != b["document_id"]


def test_search_on_empty_store_returns_empty_results(client):
    response = client.post("/search", json={"query": "anything"})

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["top_k"] == 5


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 1001},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 51},
        {"query": "ok", "top_k": "many"},
        {},
    ],
)
def test_invalid_search_requests_return_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_query_without_searchable_words_returns_422(client):
    response = client.post("/search", json={"query": "?!..."})

    assert response.status_code == 422
    assert "no searchable words" in response.json()["detail"]


def test_search_unknown_document_returns_404(client):
    response = client.post("/search", json={"query": "ok", "document_id": "missing"})

    assert response.status_code == 404


def test_get_unknown_document_returns_404(client):
    response = client.get("/documents/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found: missing"}


def test_non_pdf_extension_returns_415(client, make_pdf):
    assert upload(client, make_pdf(["text"]), "notes.txt").status_code == 415


def test_non_pdf_content_returns_415(client):
    assert upload(client, b"just some text", "fake.pdf").status_code == 415


def test_corrupt_pdf_returns_400(client):
    response = upload(client, b"%PDF-1.7\nthis is not a valid body", "broken.pdf")

    assert response.status_code == 400


def test_empty_file_returns_400(client):
    assert upload(client, b"", "empty.pdf").status_code == 400


def test_file_too_large_returns_413(client):
    assert upload(client, b"%PDF-" + b"0" * 60_000, "big.pdf").status_code == 413


def test_pdf_without_text_returns_422(client, make_pdf):
    response = upload(client, make_pdf(["", ""]), "scan.pdf")

    assert response.status_code == 422
    assert client.get("/documents").json() == []


def test_duplicate_upload_returns_409(client, make_pdf):
    data = make_pdf(["same bytes"])
    first = upload(client, data).json()

    response = upload(client, data, "copy.pdf")

    assert response.status_code == 409
    assert first["document_id"] in response.json()["detail"]


def test_missing_file_field_returns_422(client):
    assert client.post("/documents").status_code == 422
