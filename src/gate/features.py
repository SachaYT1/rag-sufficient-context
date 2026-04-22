"""Feature engineering for the selective-generation gate."""

from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "confidence",
    "sufficient",
    "positive_chunk_ratio",
    "support_title_recall_post_truncation",
)


def _feature_vector(example: dict[str, Any]) -> list[float]:
    support_recall_post = example.get("support_title_recall_post_truncation")
    support_recall_post = float(support_recall_post) if support_recall_post is not None else 0.0
    return [
        float(example.get("confidence", 0.0)),
        float(bool(example.get("sufficient", False))),
        float(example.get("positive_chunk_ratio", 0.0)),
        support_recall_post,
    ]


def prepare_features(
    examples: list[dict[str, Any]],
    include_abstentions: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extract gate features and binary labels for answerable examples.

    Labels: 1 = correct, 0 = hallucinate. Abstentions excluded by default.
    """
    rows = [ex for ex in examples if include_abstentions or ex.get("category") != "abstain"]
    features: list[list[float]] = []
    labels: list[float] = []
    for row in rows:
        features.append(_feature_vector(row))
        labels.append(float(row.get("category") == "correct"))
    return np.array(features, dtype=float), np.array(labels, dtype=float), rows
