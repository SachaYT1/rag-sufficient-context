"""Calibration metrics and post-hoc recalibration utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    if len(probs) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


def compute_calibration_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if len(scores) == 0:
        return {"ece": None, "brier": None, "auroc": None, "auprc": None}

    metrics: dict[str, Any] = {
        "ece": _compute_ece(scores, labels),
        "brier": float(brier_score_loss(labels, scores)),
        "auroc": None,
        "auprc": None,
    }
    if len(np.unique(labels)) >= 2:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["auprc"] = float(average_precision_score(labels, scores))
    return metrics


def isotonic_calibrate(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic regression calibrator."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=float))
    return iso


def platt_calibrate(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Fit a Platt-scaling (1D logistic regression) calibrator."""
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(labels, dtype=int))
    return clf
