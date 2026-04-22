"""Factory for retrievers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRRFRetriever
from src.retrieval.reranker import CrossEncoderReranker

RETRIEVER_REGISTRY: dict[str, Callable[..., BaseRetriever]] = {}


def register_retriever(name: str) -> Callable[[Callable[..., BaseRetriever]], Callable[..., BaseRetriever]]:
    def decorator(fn: Callable[..., BaseRetriever]) -> Callable[..., BaseRetriever]:
        RETRIEVER_REGISTRY[name] = fn
        return fn

    return decorator


@register_retriever("bm25")
def _build_bm25(**kwargs: Any) -> BaseRetriever:
    return BM25Retriever()


@register_retriever("dense")
def _build_dense(**kwargs: Any) -> BaseRetriever:
    return DenseRetriever(
        model_name=kwargs.get("dense_model", "intfloat/e5-base-v2"),
        device=kwargs.get("device"),
    )


@register_retriever("hybrid_rrf")
def _build_hybrid(**kwargs: Any) -> BaseRetriever:
    return HybridRRFRetriever(
        dense_model_name=kwargs.get("dense_model", "intfloat/e5-base-v2"),
        rrf_k=kwargs.get("rrf_k", 60),
        weights=tuple(kwargs.get("hybrid_weights", (0.5, 0.5))),
    )


@register_retriever("cross_encoder")
def _build_reranker(**kwargs: Any) -> BaseRetriever:
    return CrossEncoderReranker(
        model_name=kwargs.get("reranker_model", "BAAI/bge-reranker-base"),
        device=kwargs.get("device"),
    )


def build_retriever(name: str = "bm25", **kwargs: Any) -> BaseRetriever:
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(
            f"Unknown retriever '{name}'. Available: {sorted(RETRIEVER_REGISTRY)}"
        )
    return RETRIEVER_REGISTRY[name](**kwargs)
