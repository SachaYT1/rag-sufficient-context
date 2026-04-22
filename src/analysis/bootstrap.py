"""Nonparametric bootstrap for metric confidence intervals and paired tests."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _percentile_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.percentile(values, 100 * (alpha / 2)))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_metric_ci(
    per_example: list[dict[str, Any]],
    metric_fn: Callable[[list[dict[str, Any]]], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compute a percentile CI for an arbitrary metric function."""
    rng = np.random.default_rng(seed)
    n = len(per_example)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    estimates = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = [per_example[j] for j in idx]
        estimates[i] = metric_fn(sample)

    point = float(metric_fn(per_example))
    lo, hi = _percentile_ci(estimates, alpha)
    return {"mean": point, "ci_low": lo, "ci_high": hi}


def _aurc_from_scores_labels(scores: np.ndarray, labels: np.ndarray, total: int) -> float:
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    risks: list[float] = []
    covs: list[float] = []
    cum_correct = 0.0
    for i, (score, label) in enumerate(zip(s, y), start=1):
        cum_correct += float(label)
        covs.append(i / total)
        risks.append(1.0 - cum_correct / i)
    if len(covs) < 2:
        return float("nan")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(risks, covs))


def bootstrap_aurc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    total: int,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile CI for AURC using bootstrap over examples."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(scores)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        estimates[i] = _aurc_from_scores_labels(scores[idx], labels[idx], total=total)

    point = _aurc_from_scores_labels(scores, labels, total=total)
    lo, hi = _percentile_ci(estimates, alpha)
    return {"mean": point, "ci_low": lo, "ci_high": hi}


def paired_bootstrap_test(
    baseline_scores: np.ndarray,
    proposed_scores: np.ndarray,
    labels: np.ndarray,
    total: int,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test that proposed < baseline AURC.

    Returns point AURCs, their difference, and a one-sided p-value.
    Lower AURC is better.
    """
    baseline_scores = np.asarray(baseline_scores, dtype=float)
    proposed_scores = np.asarray(proposed_scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(labels)
    if n == 0:
        return {"p_value": float("nan"), "delta": 0.0}

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b = _aurc_from_scores_labels(baseline_scores[idx], labels[idx], total=total)
        p = _aurc_from_scores_labels(proposed_scores[idx], labels[idx], total=total)
        deltas[i] = p - b

    point_baseline = _aurc_from_scores_labels(baseline_scores, labels, total=total)
    point_proposed = _aurc_from_scores_labels(proposed_scores, labels, total=total)

    p_value = float((deltas >= 0).mean())  # prob(proposed not better)
    return {
        "baseline_aurc": point_baseline,
        "proposed_aurc": point_proposed,
        "delta": point_proposed - point_baseline,
        "p_value_proposed_better": p_value,
        "n_bootstrap": n_bootstrap,
    }
