import pytest

from app.ingestion.chunking import Chunk
from evaluation.corpus import load_corpus
from evaluation.run_eval import evaluate, evidence_groups, load_queries


def make_chunk(idx: int, text: str, start: int) -> Chunk:
    return Chunk(f"d:p1:c{idx}", "d", 1, text, start, start + len(text))


# Normalized page "aaaa bbbb cccc dddd" split into two chunks.
CHUNKS = {"doc.pdf": [make_chunk(0, "aaaa bbbb", 0), make_chunk(1, "cccc dddd", 10)]}


def query(evidence: str) -> dict:
    return {"id": "t", "relevant": [{"filename": "doc.pdf", "page": 1, "evidence": evidence}]}


def test_evidence_inside_one_chunk_maps_to_that_chunk():
    assert evidence_groups(query("cccc"), CHUNKS) == [{"d:p1:c1"}]


def test_evidence_split_evenly_maps_to_both_chunks():
    assert evidence_groups(query("bbbb cccc"), CHUNKS) == [{"d:p1:c0", "d:p1:c1"}]


def test_evidence_mostly_in_one_chunk_maps_to_that_chunk():
    assert evidence_groups(query("bb cccc dd"), CHUNKS) == [{"d:p1:c1"}]


def test_missing_evidence_raises():
    with pytest.raises(ValueError, match="evidence not found"):
        evidence_groups(query("zzzz"), CHUNKS)


@pytest.mark.parametrize("chunk_size", [50, 100, 200])
def test_all_labels_resolve_for_experiment_chunk_sizes(chunk_size):
    summary, results = evaluate(chunk_size, load_corpus(), load_queries())

    assert summary["answerable"] == sum(1 for r in results if r.query["relevant"])
    assert 0.0 <= summary["mrr@10"] <= 1.0
