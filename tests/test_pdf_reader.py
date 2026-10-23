import pytest

from app.errors import InvalidPDFError, NoExtractableTextError, PDFNotFoundError
from app.ingestion.pdf_reader import extract_pages


def test_extracts_pages_in_order_with_one_based_numbers(make_pdf):
    pages = extract_pages(make_pdf(["alpha page", "beta page", "gamma page"]))

    assert [p.page_number for p in pages] == [1, 2, 3]
    assert ["alpha" in pages[0].text, "beta" in pages[1].text, "gamma" in pages[2].text] == [True] * 3


def test_reads_from_path(make_pdf, tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_pdf(["hello from disk"]))

    pages = extract_pages(path)

    assert len(pages) == 1
    assert "hello from disk" in pages[0].text


def test_empty_page_is_kept_so_numbering_does_not_shift(make_pdf):
    pages = extract_pages(make_pdf(["first", "", "third"]))

    assert [p.page_number for p in pages] == [1, 2, 3]
    assert pages[1].text.strip() == ""
    assert "third" in pages[2].text


def test_pdf_without_any_text_raises(make_pdf):
    with pytest.raises(NoExtractableTextError, match="2 page"):
        extract_pages(make_pdf(["", ""]))


def test_missing_file_raises(tmp_path):
    with pytest.raises(PDFNotFoundError):
        extract_pages(tmp_path / "missing.pdf")


@pytest.mark.parametrize("data", [b"", b"this is plain text", b"%PDF-1.7 truncated garbage"])
def test_invalid_bytes_raise(data):
    with pytest.raises(InvalidPDFError):
        extract_pages(data)


def test_invalid_file_on_disk_raises(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_text("not really a pdf")

    with pytest.raises(InvalidPDFError):
        extract_pages(path)
