"""BM25 retriever built on rank_bm25."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.base import BaseRetriever


def tokenize_simple(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_bm25_index(passages: list[dict[str, Any]] | list[str]) -> BM25Okapi:
    normalized = [p["text"] if isinstance(p, dict) else p for p in passages]
    return BM25Okapi([tokenize_simple(p) for p in normalized])


class BM25Retriever(BaseRetriever):
    """Classic lexical retriever using BM25."""

    name = "bm25"

    def score(self, query: str, passages: list[dict[str, Any]]) -> list[float]:
        index = build_bm25_index(passages)
        scores = index.get_scores(tokenize_simple(query))
        return [float(s) for s in scores]
