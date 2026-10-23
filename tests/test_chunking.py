import pytest

from app.errors import InvalidParameterError
from app.ingestion.chunking import chunk_page, chunk_pages, normalize_text
from app.ingestion.pdf_reader import PageText


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_normalize_collapses_whitespace_but_keeps_punctuation_case_and_diacritics():
    raw = "  Hello,\n  World!\t Tiếng   Việt\r\n có dấu.  "

    assert normalize_text(raw) == "Hello, World! Tiếng Việt có dấu."


def test_normalize_composes_unicode_to_nfc():
    decomposed = "é"  # "e" + combining acute accent

    assert normalize_text(decomposed) == "é"


def test_short_page_gives_one_chunk():
    chunks = chunk_page("d", PageText(1, "just three words"), chunk_size=10)

    assert [c.text for c in chunks] == ["just three words"]


def test_exact_boundary_gives_full_chunks_only():
    chunks = chunk_page("d", PageText(1, words(20)), chunk_size=10)

    assert [len(c.text.split()) for c in chunks] == [10, 10]


def test_final_short_chunk_is_kept():
    chunks = chunk_page("d", PageText(1, words(25)), chunk_size=10)

    assert [len(c.text.split()) for c in chunks] == [10, 10, 5]


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_empty_page_gives_no_chunks(text):
    assert chunk_page("d", PageText(1, text), chunk_size=10) == []


def test_content_is_preserved_and_offsets_point_into_normalized_text():
    raw = "line one of the page\nline two,   with  extra   spaces\n\nand a third line"
    page = PageText(4, raw)
    normalized = normalize_text(raw)

    chunks = chunk_page("d", page, chunk_size=3)

    assert " ".join(c.text for c in chunks) == normalized
    for c in chunks:
        assert normalized[c.char_start : c.char_end] == c.text


def test_metadata_ids_and_order_across_pages():
    pages = [PageText(1, words(15, "a")), PageText(2, ""), PageText(3, words(5, "c"))]

    chunks = chunk_pages("doc1", pages, chunk_size=10)

    assert [c.chunk_id for c in chunks] == ["doc1:p1:c0", "doc1:p1:c1", "doc1:p3:c0"]
    assert [c.page_number for c in chunks] == [1, 1, 3]
    assert all(c.document_id == "doc1" for c in chunks)
    assert all(c.text for c in chunks)


def test_chunks_never_cross_pages():
    pages = [PageText(1, words(3, "a")), PageText(2, words(3, "b"))]

    chunks = chunk_pages("d", pages, chunk_size=100)

    assert [c.text for c in chunks] == ["a0 a1 a2", "b0 b1 b2"]


@pytest.mark.parametrize("size", [0, -5])
def test_invalid_chunk_size_raises(size):
    with pytest.raises(InvalidParameterError):
        chunk_page("d", PageText(1, "text"), chunk_size=size)
