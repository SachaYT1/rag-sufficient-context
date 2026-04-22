"""Factory/registry for confidence estimators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.confidence.base import BaseConfidenceEstimator
from src.confidence.estimators import (
    InlineConfidenceEstimator,
    PTrueConfidenceEstimator,
    SelfConsistencyConfidenceEstimator,
    SelfReportConfidenceEstimator,
    SemanticEntropyConfidenceEstimator,
    TokenEntropyConfidenceEstimator,
)

CONFIDENCE_REGISTRY: dict[str, Callable[..., BaseConfidenceEstimator]] = {}


def register_confidence(name: str) -> Callable[
    [Callable[..., BaseConfidenceEstimator]],
    Callable[..., BaseConfidenceEstimator],
]:
    def decorator(fn):
        CONFIDENCE_REGISTRY[name] = fn
        return fn

    return decorator


@register_confidence("inline")
def _inline(**kwargs: Any) -> BaseConfidenceEstimator:
    return InlineConfidenceEstimator()


@register_confidence("self_report")
def _self_report(**kwargs: Any) -> BaseConfidenceEstimator:
    return SelfReportConfidenceEstimator(max_new_tokens=kwargs.get("max_new_tokens", 64))


@register_confidence("token_entropy")
def _token_entropy(**kwargs: Any) -> BaseConfidenceEstimator:
    return TokenEntropyConfidenceEstimator()


@register_confidence("p_true")
def _p_true(**kwargs: Any) -> BaseConfidenceEstimator:
    return PTrueConfidenceEstimator(max_new_tokens=kwargs.get("max_new_tokens", 64))


@register_confidence("self_consistency")
def _sc(**kwargs: Any) -> BaseConfidenceEstimator:
    return SelfConsistencyConfidenceEstimator(
        num_samples=kwargs.get("num_samples", 5),
        sample_temperature=kwargs.get("sample_temperature", 0.7),
        sample_top_p=kwargs.get("sample_top_p", 0.95),
        max_new_tokens=kwargs.get("max_new_tokens", 256),
    )


@register_confidence("semantic_entropy")
def _se(**kwargs: Any) -> BaseConfidenceEstimator:
    return SemanticEntropyConfidenceEstimator(
        num_samples=kwargs.get("num_samples", 5),
        sample_temperature=kwargs.get("sample_temperature", 0.7),
        sample_top_p=kwargs.get("sample_top_p", 0.95),
        nli_model_name=kwargs.get(
            "nli_model_name", "cross-encoder/nli-deberta-v3-small"
        ),
        max_new_tokens=kwargs.get("max_new_tokens", 256),
    )


def build_confidence_estimator(
    method: str = "inline",
    config: dict[str, Any] | None = None,
) -> BaseConfidenceEstimator:
    config = config or {}
    if method not in CONFIDENCE_REGISTRY:
        raise ValueError(
            f"Unsupported confidence method: {method}. "
            f"Available: {sorted(CONFIDENCE_REGISTRY)}"
        )
    return CONFIDENCE_REGISTRY[method](**config)
