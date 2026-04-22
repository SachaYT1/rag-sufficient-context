"""BM25 determinism and basic ranking sanity."""

from __future__ import annotations

from src.retrieval.bm25 import BM25Retriever, tokenize_simple


def test_tokenize_lowercase() -> None:
    assert tokenize_simple("Hello, WORLD!") == ["hello", "world"]


def test_bm25_ranks_match_query() -> None:
    passages = [
        {"text": "The Eiffel Tower is in Paris, France."},
        {"text": "The Great Wall is a famous Chinese landmark."},
        {"text": "Bananas are yellow fruits."},
    ]
    retriever = BM25Retriever()
    scores = retriever.score("Where is the Eiffel Tower?", passages)
    assert len(scores) == 3
    assert scores[0] >= scores[1] and scores[0] >= scores[2]


def test_bm25_deterministic() -> None:
    passages = [{"text": f"doc {i} with shared word token"} for i in range(5)]
    r1 = BM25Retriever().score("shared token", passages)
    r2 = BM25Retriever().score("shared token", passages)
    assert r1 == r2
