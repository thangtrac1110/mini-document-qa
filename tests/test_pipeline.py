import pytest

from app.config import Settings
from app.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidPDFError,
    NoExtractableTextError,
)
from app.pipeline import DocumentService


@pytest.fixture
def service():
    return DocumentService(Settings(chunk_size_words=20))


def test_ingest_then_search_returns_correct_document_and_page(service, make_pdf):
    doc = service.ingest_pdf(
        "astro.pdf",
        make_pdf(["The Sun is a star.", "Jupiter is the largest planet in the solar system."]),
    )

    hits = service.search("largest planet", top_k=3)

    assert doc.page_count == 2
    assert doc.chunk_count == 2
    assert hits[0].document_id == doc.document_id
    assert hits[0].filename == "astro.pdf"
    assert hits[0].page_number == 2
    assert "Jupiter" in hits[0].text


def test_multiple_documents_and_document_filter(service, make_pdf):
    bikes = service.ingest_pdf("bikes.pdf", make_pdf(["Inflate bicycle tyres to the pressure on the sidewall."]))
    cars = service.ingest_pdf("cars.pdf", make_pdf(["Check car tyres pressure monthly when cold."]))

    all_hits = service.search("tyres pressure", top_k=10)
    car_hits = service.search("tyres pressure", top_k=10, document_id=cars.document_id)

    assert {h.document_id for h in all_hits} == {bikes.document_id, cars.document_id}
    assert [h.document_id for h in car_hits] == [cars.document_id]
    assert [d.filename for d in service.list_documents()] == ["bikes.pdf", "cars.pdf"]


def test_duplicate_upload_is_rejected(service, make_pdf):
    data = make_pdf(["same content"])
    first = service.ingest_pdf("a.pdf", data)

    with pytest.raises(DuplicateDocumentError) as excinfo:
        service.ingest_pdf("renamed.pdf", data)

    assert excinfo.value.existing_document_id == first.document_id
    assert len(service.list_documents()) == 1


def test_invalid_pdf_leaves_state_unchanged(service, make_pdf):
    service.ingest_pdf("ok.pdf", make_pdf(["valid searchable text"]))

    with pytest.raises(InvalidPDFError):
        service.ingest_pdf("bad.pdf", b"garbage")

    assert [d.filename for d in service.list_documents()] == ["ok.pdf"]
    assert len(service.search("valid", top_k=5)) == 1


def test_pdf_without_text_is_not_registered(service, make_pdf):
    with pytest.raises(NoExtractableTextError):
        service.ingest_pdf("scan.pdf", make_pdf(["", ""]))

    assert service.list_documents() == []


def test_indexing_failure_does_not_mark_document_ready(service, make_pdf, monkeypatch):
    service.ingest_pdf("first.pdf", make_pdf(["existing searchable text"]))

    def broken_index(chunks):
        raise RuntimeError("index build failed")

    monkeypatch.setattr(service, "_build_index", broken_index)
    with pytest.raises(RuntimeError):
        service.ingest_pdf("second.pdf", make_pdf(["new text that should not appear"]))

    assert [d.filename for d in service.list_documents()] == ["first.pdf"]
    assert service.search("appear", top_k=5) == []
    assert len(service.search("existing", top_k=5)) == 1


def test_unknown_document_raises(service):
    with pytest.raises(DocumentNotFoundError):
        service.get_document("nope")
    with pytest.raises(DocumentNotFoundError):
        service.search("anything", top_k=5, document_id="nope")


def test_search_with_no_documents_returns_empty(service):
    assert service.search("anything", top_k=5) == []
