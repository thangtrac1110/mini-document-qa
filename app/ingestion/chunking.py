"""Text normalization and fixed-size, page-bounded chunking.

Algorithm:
1. ``normalize_text``: Unicode NFC, then collapse every whitespace run (spaces, line
   breaks from PDF layout, tabs) to a single space and strip the ends. Punctuation, case
   and diacritics (e.g. Vietnamese) are preserved.
2. ``chunk_page``: split the normalized page into consecutive groups of ``chunk_size``
   whitespace-separated words, with no overlap. A chunk never crosses a page boundary.
   The final chunk of a page may be shorter than ``chunk_size``; it is kept as is.
   An empty page produces no chunks.

Content-preservation property: for every page,
``" ".join(c.text for c in chunks) == normalize_text(page.text)``, and each chunk's
``text == normalized_page[char_start:char_end]``.
"""

import re
import unicodedata
from dataclasses import dataclass

from app.errors import InvalidParameterError
from app.ingestion.pdf_reader import PageText

_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<document_id>:p<page>:c<index within page>"
    document_id: str
    page_number: int
    text: str
    char_start: int  # offsets into the normalized page text
    char_end: int


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def chunk_page(document_id: str, page: PageText, chunk_size: int) -> list[Chunk]:
    if chunk_size < 1:
        raise InvalidParameterError(f"chunk_size must be >= 1, got {chunk_size}")

    text = normalize_text(page.text)
    words = list(_WORD.finditer(text))
    chunks = []
    for index, start in enumerate(range(0, len(words), chunk_size)):
        group = words[start : start + chunk_size]
        char_start, char_end = group[0].start(), group[-1].end()
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}:p{page.page_number}:c{index}",
                document_id=document_id,
                page_number=page.page_number,
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
            )
        )
    return chunks


def chunk_pages(document_id: str, pages: list[PageText], chunk_size: int) -> list[Chunk]:
    return [chunk for page in pages for chunk in chunk_page(document_id, page, chunk_size)]
