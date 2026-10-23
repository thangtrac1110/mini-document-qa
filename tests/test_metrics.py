import pytest

from evaluation.metrics import mean, precision_at_k, recall_at_k, reciprocal_rank

# Hand-calculated example.
# Ranking:          a  b  c  d  e
# Evidence groups:  {b}, {d, x}, {z}   (x and z were not retrieved)
RANKED = ["a", "b", "c", "d", "e"]
GROUPS = [{"b"}, {"d", "x"}, {"z"}]


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (1, 0 / 3),  # top-1 = {a}: no group hit
        (2, 1 / 3),  # {a, b}: group {b} hit
        (4, 2 / 3),  # {a, b, c, d}: {b} and {d, x} hit
        (5, 2 / 3),  # {z} is never retrieved
    ],
)
def test_recall_at_k(k, expected):
    assert recall_at_k(RANKED, GROUPS, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (1, 0 / 1),
        (2, 1 / 2),  # b
        (4, 2 / 4),  # b, d
        (5, 2 / 5),
        (10, 2 / 10),  # denominator stays k when fewer than k results exist
    ],
)
def test_precision_at_k(k, expected):
    assert precision_at_k(RANKED, GROUPS, k) == pytest.approx(expected)


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(RANKED, GROUPS) == pytest.approx(1 / 2)


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_retrieved():
    assert reciprocal_rank(["a", "c"], GROUPS) == 0.0
    assert reciprocal_rank([], GROUPS) == 0.0


def test_two_chunks_of_one_passage_count_once_for_recall_but_twice_for_precision():
    groups = [{"p1", "p2"}]

    assert recall_at_k(["p1", "p2"], groups, 2) == 1.0
    assert precision_at_k(["p1", "p2"], groups, 2) == 1.0


def test_mrr_over_queries():
    assert mean([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert mean([]) == 0.0


def test_metrics_reject_queries_without_labels_and_bad_k():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, [], 5)
    with pytest.raises(ValueError):
        precision_at_k(RANKED, GROUPS, 0)
