import pytest

from app.errors import EmptyQueryError, InvalidParameterError
from app.ingestion.chunking import Chunk
from app.retrieval.bm25_index import BM25Index


def make_chunk(doc: str, page: int, idx: int, text: str) -> Chunk:
    return Chunk(f"{doc}:p{page}:c{idx}", doc, page, text, 0, len(text))


CORPUS = [
    make_chunk("garden", 1, 0, "Tomatoes need full sun and regular watering to grow well."),
    make_chunk("garden", 2, 0, "Compost improves soil structure and feeds earthworms."),
    make_chunk("space", 1, 0, "The Moon orbits the Earth roughly every 27 days."),
    make_chunk("space", 2, 0, "Mars has two small moons named Phobos and Deimos."),
    make_chunk("cooking", 1, 0, "Simmer the tomato sauce slowly with garlic and basil."),
]


@pytest.fixture
def index():
    return BM25Index(CORPUS)


def test_most_relevant_chunk_ranks_first(index):
    results = index.search("moons of Mars", top_k=3)

    assert results[0].chunk.chunk_id == "space:p2:c0"


def test_distinctive_term_finds_its_chunk(index):
    results = index.search("compost for the soil", top_k=5)

    assert results[0].chunk.chunk_id == "garden:p2:c0"


def test_scores_are_sorted_descending(index):
    results = index.search("tomatoes tomato sauce sun", top_k=5)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk.chunk_id == "cooking:p1:c0"  # matches "tomato" and "sauce"


def test_results_respect_top_k(index):
    assert len(index.search("the", top_k=2)) == 2


def test_chunks_without_overlap_are_not_returned(index):
    results = index.search("earthworms", top_k=5)

    assert [r.chunk.chunk_id for r in results] == ["garden:p2:c0"]


def test_unknown_terms_return_nothing(index):
    assert index.search("quantum entanglement", top_k=5) == []


def test_document_filter(index):
    results = index.search("the", top_k=10, document_id="space")

    assert results
    assert {r.chunk.document_id for r in results} == {"space"}


def test_results_carry_original_metadata(index):
    result = index.search("Phobos", top_k=1)[0]

    assert result.chunk == CORPUS[3]


@pytest.mark.parametrize("query", ["", "   ", "?!..."])
def test_empty_query_raises(index, query):
    with pytest.raises(EmptyQueryError):
        index.search(query, top_k=3)


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises(index, top_k):
    with pytest.raises(InvalidParameterError):
        index.search("moon", top_k=top_k)


def test_empty_index_returns_no_results():
    assert BM25Index([]).search("anything", top_k=5) == []


def test_empty_query_raises_even_on_empty_index():
    with pytest.raises(EmptyQueryError):
        BM25Index([]).search("  ", top_k=5)


def test_single_and_two_chunk_corpora_still_score_matches():
    # Regression: rank_bm25's default IDF gives 0 (N=2, n=1) or negative (N=1) weights here.
    single = BM25Index([CORPUS[0]])
    pair = BM25Index(CORPUS[:2])

    assert single.search("tomatoes", top_k=1)[0].score > 0
    assert pair.search("compost", top_k=1)[0].chunk.chunk_id == "garden:p2:c0"
