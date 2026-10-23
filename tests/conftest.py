import pymupdf
import pytest


def build_pdf(pages: list[str]) -> bytes:
    """Build an in-memory PDF; each string becomes one page ("" gives a blank page)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(72, 72, 540, 770), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


class FakeLLMClient:
    """Stands in for the provider: returns a canned reply (or raises) and records calls."""

    def __init__(self, reply: dict | Exception | None = None):
        self.reply = reply if reply is not None else {
            "answer": "fake answer",
            "supporting_source_ids": ["S1"],
            "insufficient_information": False,
        }
        self.calls: list[dict] = []

    def complete_json(self, *, system: str, user: str, schema: dict) -> dict:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def make_pdf():
    return build_pdf


@pytest.fixture
def fake_llm():
    return FakeLLMClient
