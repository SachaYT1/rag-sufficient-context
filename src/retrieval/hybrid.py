"""Hybrid retrievers: Reciprocal Rank Fusion over BM25 and Dense."""

from __future__ import annotations

from typing import Any

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever


def _rrf_from_ranks(ranks: list[int], k: int = 60) -> list[float]:
    return [1.0 / (k + r) for r in ranks]


def _scores_to_ranks(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rank_of = [0] * len(scores)
    for rank, idx in enumerate(order):
        rank_of[idx] = rank + 1
    return rank_of


class HybridRRFRetriever(BaseRetriever):
    """Reciprocal Rank Fusion of BM25 and dense retrievers."""

    name = "hybrid_rrf"

    def __init__(
        self,
        dense_model_name: str = "intfloat/e5-base-v2",
        rrf_k: int = 60,
        weights: tuple[float, float] = (0.5, 0.5),
    ):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(model_name=dense_model_name)
        self.rrf_k = rrf_k
        self.weights = weights

    def score(self, query: str, passages: list[dict[str, Any]]) -> list[float]:
        bm25_scores = self.bm25.score(query, passages)
        dense_scores = self.dense.score(query, passages)
        bm25_rrf = _rrf_from_ranks(_scores_to_ranks(bm25_scores), k=self.rrf_k)
        dense_rrf = _rrf_from_ranks(_scores_to_ranks(dense_scores), k=self.rrf_k)
        w_bm25, w_dense = self.weights
        return [w_bm25 * a + w_dense * b for a, b in zip(bm25_rrf, dense_rrf, strict=False)]
