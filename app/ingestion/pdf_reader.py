"""Page-by-page text extraction from text-based PDFs using PyMuPDF.

Contract:
- Page numbers are 1-based and returned in document order.
- A page without extractable text is kept, with ``text == ""``, so numbering never shifts.
- If *no* page has extractable text, ``NoExtractableTextError`` is raised. This is the
  typical outcome for scanned PDFs; OCR is out of scope.
- Parse failures raise ``InvalidPDFError``; they are never turned into empty text.
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.errors import InvalidPDFError, NoExtractableTextError, PDFNotFoundError


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based
    text: str


def extract_pages(source: str | Path | bytes) -> list[PageText]:
    """Extract raw text for every page of a PDF given as a path or as bytes."""
    doc = _open(source)
    try:
        if doc.needs_pass:
            raise InvalidPDFError("Encrypted PDFs are not supported")
        if doc.page_count == 0:
            raise InvalidPDFError("PDF has no pages")
        pages = [PageText(page_number=i + 1, text=page.get_text("text")) for i, page in enumerate(doc)]
    finally:
        doc.close()

    if not any(p.text.strip() for p in pages):
        raise NoExtractableTextError(
            f"No extractable text found in {len(pages)} page(s); "
            "the PDF may be scanned (OCR is not supported)"
        )
    return pages


def _open(source: str | Path | bytes) -> pymupdf.Document:
    if isinstance(source, bytes):
        try:
            return pymupdf.open(stream=source, filetype="pdf")
        except pymupdf.FileDataError as exc:  # includes EmptyFileError
            raise InvalidPDFError(f"Not a readable PDF: {exc}") from exc

    path = Path(source)
    if not path.is_file():
        raise PDFNotFoundError(f"File not found: {path}")
    try:
        return pymupdf.open(path, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise InvalidPDFError(f"Not a readable PDF: {path.name}: {exc}") from exc
