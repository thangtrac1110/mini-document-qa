"""Evaluation corpus: original text files rendered to PDFs in memory.

Each ``data/corpus/<name>.txt`` file becomes ``<name>.pdf``; pages are separated by a
line containing only ``---page---``. Rendering happens at run time, so no binary PDFs
are stored in the repository.

Write the PDFs to disk for manual testing with:
    python -m evaluation.corpus samples/
"""

import sys
from pathlib import Path

import pymupdf

CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
PAGE_BREAK = "\n---page---\n"


def build_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()  # A4-like default page size
        overflow = page.insert_textbox(pymupdf.Rect(60, 60, 535, 780), text, fontsize=10.5)
        if overflow < 0:
            raise ValueError(f"Page text does not fit on one PDF page: {text[:60]!r}...")
    data = doc.tobytes()
    doc.close()
    return data


def load_corpus() -> dict[str, bytes]:
    """Return ``{filename: pdf_bytes}`` for every corpus text file, sorted by name."""
    corpus = {}
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        pages = path.read_text(encoding="utf-8").split(PAGE_BREAK)
        corpus[path.with_suffix(".pdf").name] = build_pdf(pages)
    return corpus


if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in load_corpus().items():
        (out_dir / name).write_bytes(data)
        print(f"wrote {out_dir / name}")
