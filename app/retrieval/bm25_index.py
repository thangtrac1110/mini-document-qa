"""BM25 ranking over chunks, built on ``rank_bm25``.

Behavior:
- Queries with no tokens raise ``EmptyQueryError``.
- An empty index returns no results.
- Only chunks sharing at least one query token (score > 0) are returned, best first;
  ties keep corpus order.
- Scores are relative ranking values for *this* corpus and query. They are not
  probabilities and are not comparable across corpora or configurations.
- The index is immutable; adding documents means building a new index (rank_bm25 has
  no incremental update).
"""

import math
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.errors import EmptyQueryError, InvalidParameterError
from app.ingestion.chunking import Chunk
from app.retrieval.tokenizer import tokenize


class _PositiveIdfBM25(BM25Okapi):
    """BM25Okapi with Lucene's IDF: log(1 + (N - n + 0.5) / (n + 0.5)), which is always > 0.

    rank_bm25's default IDF, log((N - n + 0.5) / (n + 0.5)), is 0 for a term found in one
    of two chunks and negative for a one-chunk corpus, so a newly ingested short PDF would
    rank nothing (see test_single_and_two_chunk_corpora_still_score_matches).
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        tokenized = [tokenize(c.text) for c in self.chunks]
        # rank_bm25 divides by the average chunk length, so skip building when there is
        # nothing to score.
        self._bm25 = _PositiveIdfBM25(tokenized, k1=k1, b=b) if any(tokenized) else None

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int, document_id: str | None = None) -> list[ScoredChunk]:
        if top_k < 1:
            raise InvalidParameterError(f"top_k must be >= 1, got {top_k}")
        query_tokens = tokenize(query)
        if not query_tokens:
            raise EmptyQueryError("Query contains no searchable words")
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query_tokens)
        candidates = [
            ScoredChunk(chunk, float(score))
            for chunk, score in zip(self.chunks, scores)
            if score > 0 and (document_id is None or chunk.document_id == document_id)
        ]
        candidates.sort(key=lambda r: r.score, reverse=True)  # stable: ties keep corpus order
        return candidates[:top_k]
