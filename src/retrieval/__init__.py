"""Retrieval subsystem with lazy heavy dependencies."""

from __future__ import annotations

from typing import Any

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25 import BM25Retriever, build_bm25_index, tokenize_simple
from src.retrieval.pipeline import (
    build_retrieval_pipeline,
    retrieve_context,
    summarize_retrieval_metrics,
)
from src.retrieval.registry import RETRIEVER_REGISTRY, build_retriever, register_retriever


def __getattr__(name: str) -> Any:
    """Lazy accessors for components that pull in heavy deps (datasets, torch)."""
    if name == "load_hotpotqa":
        from src.retrieval.hotpotqa import load_hotpotqa

        return load_hotpotqa
    if name == "DenseRetriever":
        from src.retrieval.dense import DenseRetriever

        return DenseRetriever
    if name == "HybridRRFRetriever":
        from src.retrieval.hybrid import HybridRRFRetriever

        return HybridRRFRetriever
    if name == "CrossEncoderReranker":
        from src.retrieval.reranker import CrossEncoderReranker

        return CrossEncoderReranker
    raise AttributeError(f"module 'src.retrieval' has no attribute {name!r}")


__all__ = [
    "BM25Retriever",
    "BaseRetriever",
    "CrossEncoderReranker",
    "DenseRetriever",
    "HybridRRFRetriever",
    "RETRIEVER_REGISTRY",
    "build_bm25_index",
    "build_retrieval_pipeline",
    "build_retriever",
    "load_hotpotqa",
    "register_retriever",
    "retrieve_context",
    "summarize_retrieval_metrics",
    "tokenize_simple",
]
