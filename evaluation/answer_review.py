"""Answer-quality review for /ask, evaluated separately from retrieval.

    python -m evaluation.answer_review run            # calls the real LLM; needs ANTHROPIC_API_KEY
    python -m evaluation.answer_review summarize evaluation/results/answer_review.csv

``run`` asks a representative subset of the labeled queries and writes a CSV with
automatic signals and empty rubric columns for a human reviewer. ``summarize`` averages
the rubric columns after review. Automatic signals are proxies, not quality scores:

- retrieval_hit: a chunk holding the labeled evidence was among the passages sent to the
  model. If "no" on an answerable query, a wrong answer is a *retrieval* failure.
- cited_evidence: some validated citation points at a labeled-evidence chunk.
- failure_hint: retrieval_failure / wrongly_abstained / answered_without_evidence.

Rubric (score each row 1, 0.5 or 0; leave blank when not applicable):
- correctness:       the answer matches the source documents.
- groundedness:      every important claim is supported by the retrieved text.
- citation_accuracy: every cited source supports the claim it is cited for.
- abstention:        no-answer query -> 1 if the system said it lacks information;
                     answerable query with retrieval_hit=yes -> 1 if it did answer.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.config import Settings
from app.generation.answer_generator import AnswerGenerator
from app.generation.llm_client import AnthropicLLMClient, LLMClient
from app.ingestion.chunking import Chunk
from app.services.qa_service import QAService
from evaluation.corpus import load_corpus
from evaluation.run_eval import build_service, evidence_groups, load_queries

DEFAULT_OUT = Path(__file__).parent / "results" / "answer_review.csv"
TOP_K = 5
# Representative subset covering every query type: 3 keyword, 4 paraphrase, 4 detail,
# 2 ambiguous, 3 no-answer.
DEFAULT_IDS = [
    "q01", "q04", "q06",
    "q07", "q08", "q11", "q12",
    "q15", "q16", "q18", "q19",
    "q28", "q29",
    "q24", "q25", "q27",
]
RUBRIC = ["correctness", "groundedness", "citation_accuracy", "abstention"]
COLUMNS = [
    "id", "type", "question", "retrieval_hit", "answer", "insufficient_information", "grounded",
    "citations", "cited_evidence", "rejected_source_ids", "failure_hint", *RUBRIC, "notes",
]


def yes_no(value: bool | None) -> str:
    return "n/a" if value is None else ("yes" if value else "no")


def failure_hint(answerable: bool, retrieval_hit: bool, abstained: bool) -> str:
    if answerable and not retrieval_hit:
        return "retrieval_failure"
    if answerable and abstained:
        return "wrongly_abstained"
    if not answerable and not abstained:
        return "answered_without_evidence"
    return ""


def review_rows(qa: QAService, chunks_by_file: dict[str, list[Chunk]], queries: list[dict]) -> list[dict]:
    rows = []
    for q in queries:
        groups = evidence_groups(q, chunks_by_file)
        relevant = set().union(*groups) if groups else set()
        result = qa.ask(q["query"], top_k=TOP_K)
        by_id = {s.source_id: s.hit for s in result.sources}
        answerable = bool(groups)
        retrieval_hit = any(s.hit.chunk_id in relevant for s in result.sources) if answerable else None
        cited_evidence = (
            any(by_id[sid].chunk_id in relevant for sid in result.cited_source_ids) if answerable else None
        )
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["query"],
            "retrieval_hit": yes_no(retrieval_hit),
            "answer": result.answer,
            "insufficient_information": yes_no(result.insufficient_information),
            "grounded": yes_no(result.grounded),
            "citations": "; ".join(
                f"{sid} {by_id[sid].filename} p{by_id[sid].page_number}" for sid in result.cited_source_ids
            ),
            "cited_evidence": yes_no(cited_evidence),
            "rejected_source_ids": " ".join(result.rejected_source_ids),
            "failure_hint": failure_hint(answerable, bool(retrieval_hit), result.insufficient_information),
            **{col: "" for col in RUBRIC},
            "notes": "",
        })
    return rows


def automatic_summary(rows: list[dict]) -> dict[str, str]:
    def rate(subset: list[dict], key: str) -> str:
        return f"{sum(r[key] == 'yes' for r in subset)}/{len(subset)}" if subset else "0/0"

    answerable = [r for r in rows if r["retrieval_hit"] != "n/a"]
    retrieved = [r for r in answerable if r["retrieval_hit"] == "yes"]
    no_answer = [r for r in rows if r["retrieval_hit"] == "n/a"]
    return {
        "retrieval hit (answerable)": rate(answerable, "retrieval_hit"),
        "cited labeled evidence (given retrieval hit)": rate(retrieved, "cited_evidence"),
        "abstained on no-answer queries": rate(no_answer, "insufficient_information"),
        "answers with rejected source IDs": f"{sum(bool(r['rejected_source_ids']) for r in rows)}/{len(rows)}",
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(path: Path) -> dict[str, tuple[float | None, int]]:
    """Mean and count of each rubric column, ignoring blank and n/a cells."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = {}
    for col in RUBRIC:
        scores = [float(r[col]) for r in rows if r[col].strip() and r[col].strip().lower() != "n/a"]
        summary[col] = (sum(scores) / len(scores) if scores else None, len(scores))
    return summary


def run(llm: LLMClient, ids: list[str]) -> list[dict]:
    wanted = set(ids)
    queries = [q for q in load_queries() if q["id"] in wanted]
    service, chunks_by_file = build_service(Settings().chunk_size_words, load_corpus())
    return review_rows(QAService(service, AnswerGenerator(llm)), chunks_by_file, queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="ask the subset with the real LLM and write a review CSV")
    run_p.add_argument("--ids", nargs="+", default=DEFAULT_IDS)
    run_p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    sum_p = sub.add_parser("summarize", help="average the rubric columns of a reviewed CSV")
    sum_p.add_argument("csv", type=Path)
    args = parser.parse_args()

    if args.command == "summarize":
        for col, (mean, n) in summarize(args.csv).items():
            print(f"{col:>18}: {'-' if mean is None else f'{mean:.2f}'} (n={n})")
        return

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY is not set; this command calls the real LLM API.")
    settings = Settings.from_env()
    rows = run(AnthropicLLMClient(api_key, settings), args.ids)
    write_csv(rows, args.out)
    print(f"Model: {settings.llm_model} (effort {settings.llm_effort}), top_k={TOP_K}, {len(rows)} queries")
    for name, value in automatic_summary(rows).items():
        print(f"{name:>45}: {value}")
    print(f"\nReview CSV written to {args.out}; fill in the rubric columns, then run `summarize`.")


if __name__ == "__main__":
    main()
