"""Retrieval metrics over a ranked list of chunk IDs.

Relevance is given as *groups*: one group per labeled evidence passage, containing the
chunk IDs that count as retrieving that passage. This keeps the metrics comparable when
chunk size changes (one passage may map to one or two chunks). With singleton groups the
definitions reduce to the standard ones.

- Recall@k:    fraction of evidence groups with at least one chunk in the top k.
- Precision@k: fraction of the top k *positions* holding a relevant chunk (denominator is
               always k, even if fewer than k results were returned).
- RR:          1 / rank of the first relevant chunk, or 0 if none is retrieved. MRR is the
               mean of RR over queries.
"""

from collections.abc import Sequence


def _check(groups: Sequence[set[str]], k: int | None = None) -> None:
    if not groups:
        raise ValueError("Metrics are undefined for a query without relevant passages")
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_at_k(ranked: Sequence[str], groups: Sequence[set[str]], k: int) -> float:
    _check(groups, k)
    top = set(ranked[:k])
    return sum(1 for g in groups if g & top) / len(groups)


def precision_at_k(ranked: Sequence[str], groups: Sequence[set[str]], k: int) -> float:
    _check(groups, k)
    relevant = set().union(*groups)
    return sum(1 for chunk_id in ranked[:k] if chunk_id in relevant) / k


def reciprocal_rank(ranked: Sequence[str], groups: Sequence[set[str]]) -> float:
    _check(groups)
    relevant = set().union(*groups)
    return next((1 / rank for rank, chunk_id in enumerate(ranked, 1) if chunk_id in relevant), 0.0)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
