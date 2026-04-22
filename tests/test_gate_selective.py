"""Synthetic sanity tests for selective-generation curves and conformal."""

from __future__ import annotations

import numpy as np

from src.gate import (
    compute_selective_curves,
    conformal_selective,
    conformal_threshold,
    prepare_features,
)


def _make_examples(n: int = 100, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    examples: list[dict] = []
    for _ in range(n):
        sufficient = bool(rng.random() < 0.6)
        confidence = float(rng.beta(5, 2) if sufficient else rng.beta(2, 5))
        is_correct = rng.random() < (0.85 if sufficient else 0.25)
        category = "correct" if is_correct else "hallucinate"
        examples.append(
            {
                "question": "q",
                "confidence": confidence,
                "sufficient": sufficient,
                "positive_chunk_ratio": float(sufficient),
                "support_title_recall_post_truncation": float(sufficient),
                "category": category,
                "answer": "a",
                "prediction": "a" if is_correct else "b",
            }
        )
    return examples


def test_features_shape() -> None:
    examples = _make_examples(n=40)
    X, y, rows = prepare_features(examples)
    assert X.shape == (len(rows), 4)
    assert set(np.unique(y)).issubset({0.0, 1.0})


def test_selective_curves_have_points() -> None:
    examples = _make_examples(n=120, seed=7)
    curves = compute_selective_curves(examples)
    assert curves["baseline"]["aurc"] is not None
    assert curves["proposed"]["aurc"] is not None
    assert len(curves["baseline"]["coverages_total"]) >= 2
    assert curves["gate_meta"]["status"] in {"cross_validated", "train_only"}


def test_conformal_threshold_monotone() -> None:
    scores = np.linspace(0, 1, 50)
    labels = (scores > 0.3).astype(float)
    tau_strict = conformal_threshold(scores, labels, alpha=0.05)
    tau_loose = conformal_threshold(scores, labels, alpha=0.3)
    assert tau_strict >= tau_loose


def test_conformal_selective_reports() -> None:
    rng = np.random.default_rng(42)
    scores = rng.random(200)
    labels = (scores + rng.normal(0, 0.1, 200) > 0.5).astype(float)
    result = conformal_selective(scores, labels, alpha=0.2, calib_fraction=0.3)
    assert "threshold" in result
    assert 0 <= result["coverage_total"] <= 1
