"""Split-conformal selective prediction with risk guarantees."""

from __future__ import annotations

from typing import Any

import numpy as np


def _split_indices(n: int, calib_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_calib = max(1, int(round(calib_fraction * n)))
    return idx[:n_calib], idx[n_calib:]


def conformal_threshold(
    calib_scores: np.ndarray,
    calib_labels: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Return the score threshold guaranteeing empirical risk <= alpha.

    We treat labels as {1 = correct, 0 = hallucinate}. Nonconformity is
    (1 - score) when the example is correct; thresholding by score means
    answer iff score >= tau. We sweep tau to the smallest value whose
    selective error on the calibration fold is <= alpha.
    """
    calib_scores = np.asarray(calib_scores, dtype=float)
    calib_labels = np.asarray(calib_labels, dtype=float)

    if len(calib_scores) == 0:
        return 1.0  # refuse everything if we have no calibration data

    ordered = np.sort(np.unique(calib_scores))[::-1]
    best_tau = float(ordered[0])
    for tau in ordered:
        mask = calib_scores >= tau
        if not mask.any():
            continue
        risk = 1.0 - float(calib_labels[mask].mean())
        if risk <= alpha:
            best_tau = float(tau)
        else:
            break
    return best_tau


def conformal_selective(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.1,
    calib_fraction: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """Split-conformal selective prediction with empirical risk guarantee.

    Returns calibration threshold and realized metrics on the test split.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(scores)

    if n == 0:
        return {
            "alpha": alpha,
            "threshold": None,
            "coverage_total": 0.0,
            "risk": None,
            "num_calib": 0,
            "num_test": 0,
        }

    calib_idx, test_idx = _split_indices(n, calib_fraction=calib_fraction, seed=seed)
    tau = conformal_threshold(
        calib_scores=scores[calib_idx],
        calib_labels=labels[calib_idx],
        alpha=alpha,
    )

    test_scores = scores[test_idx]
    test_labels = labels[test_idx]
    mask = test_scores >= tau
    kept = int(mask.sum())
    coverage = kept / max(1, len(test_idx))
    risk = 1.0 - float(test_labels[mask].mean()) if kept > 0 else None

    return {
        "alpha": alpha,
        "threshold": tau,
        "coverage_total": coverage,
        "risk": risk,
        "num_calib": int(len(calib_idx)),
        "num_test": int(len(test_idx)),
    }
