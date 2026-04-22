"""Confidence estimation subsystem."""

from __future__ import annotations

from src.confidence.base import BaseConfidenceEstimator, parse_probability_response
from src.confidence.estimators import (
    InlineConfidenceEstimator,
    PTrueConfidenceEstimator,
    SelfConsistencyConfidenceEstimator,
    SelfReportConfidenceEstimator,
    SemanticEntropyConfidenceEstimator,
    TokenEntropyConfidenceEstimator,
)
from src.confidence.pipeline import estimate_confidence_batch, estimate_confidence_ensemble
from src.confidence.registry import (
    CONFIDENCE_REGISTRY,
    build_confidence_estimator,
    register_confidence,
)

__all__ = [
    "BaseConfidenceEstimator",
    "CONFIDENCE_REGISTRY",
    "InlineConfidenceEstimator",
    "PTrueConfidenceEstimator",
    "SelfConsistencyConfidenceEstimator",
    "SelfReportConfidenceEstimator",
    "SemanticEntropyConfidenceEstimator",
    "TokenEntropyConfidenceEstimator",
    "build_confidence_estimator",
    "estimate_confidence_batch",
    "estimate_confidence_ensemble",
    "parse_probability_response",
    "register_confidence",
]
