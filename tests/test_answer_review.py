import csv

from evaluation.answer_review import (
    DEFAULT_IDS,
    automatic_summary,
    failure_hint,
    run,
    summarize,
    write_csv,
)
from evaluation.run_eval import load_queries


def test_default_subset_exists_and_covers_every_query_type():
    queries = {q["id"]: q for q in load_queries()}

    assert set(DEFAULT_IDS) <= set(queries)
    assert {queries[i]["type"] for i in DEFAULT_IDS} == {q["type"] for q in queries.values()}


def test_failure_hints_separate_retrieval_from_generation():
    assert failure_hint(answerable=True, retrieval_hit=False, abstained=False) == "retrieval_failure"
    assert failure_hint(answerable=True, retrieval_hit=True, abstained=True) == "wrongly_abstained"
    assert failure_hint(answerable=False, retrieval_hit=False, abstained=False) == "answered_without_evidence"
    assert failure_hint(answerable=True, retrieval_hit=True, abstained=False) == ""
    assert failure_hint(answerable=False, retrieval_hit=False, abstained=True) == ""


def test_run_with_fake_llm_produces_rows_with_automatic_signals(fake_llm):
    llm = fake_llm({"answer": "a", "supporting_source_ids": ["S1"], "insufficient_information": False})

    rows = {r["id"]: r for r in run(llm, ["q16", "q26"])}

    # q16: keyword-rich detail question; BM25 ranks the evidence chunk first, the fake cites S1.
    assert rows["q16"]["retrieval_hit"] == "yes"
    assert rows["q16"]["cited_evidence"] == "yes"
    assert "lighthouses.pdf p3" in rows["q16"]["citations"]
    # q26: no-answer query with no lexical overlap -> no passages, LLM not called.
    assert rows["q26"]["retrieval_hit"] == "n/a"
    assert rows["q26"]["insufficient_information"] == "yes"
    assert len(llm.calls) == 1


def test_automatic_summary_counts():
    rows = [
        {"retrieval_hit": "yes", "cited_evidence": "yes", "insufficient_information": "no", "rejected_source_ids": ""},
        {"retrieval_hit": "no", "cited_evidence": "no", "insufficient_information": "no", "rejected_source_ids": "S9"},
        {"retrieval_hit": "n/a", "cited_evidence": "n/a", "insufficient_information": "yes", "rejected_source_ids": ""},
    ]

    summary = automatic_summary(rows)

    assert summary["retrieval hit (answerable)"] == "1/2"
    assert summary["cited labeled evidence (given retrieval hit)"] == "1/1"
    assert summary["abstained on no-answer queries"] == "1/1"
    assert summary["answers with rejected source IDs"] == "1/3"


def test_summarize_ignores_blank_and_na_cells(tmp_path, fake_llm):
    path = tmp_path / "review.csv"
    rows = run(fake_llm(), ["q01", "q04"])
    rows[0].update(correctness="1", groundedness="0.5", citation_accuracy="n/a", abstention="1")
    rows[1].update(correctness="0", groundedness="", citation_accuracy="", abstention="1")
    write_csv(rows, path)

    summary = summarize(path)

    assert summary["correctness"] == (0.5, 2)
    assert summary["groundedness"] == (0.5, 1)
    assert summary["citation_accuracy"] == (None, 0)
    assert summary["abstention"] == (1.0, 2)
    with path.open(encoding="utf-8") as f:
        assert next(csv.reader(f))[0] == "id"
