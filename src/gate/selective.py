"""Selective-generation curves and coverage/risk analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.gate.calibration import compute_calibration_metrics
from src.gate.features import prepare_features
from src.gate.models import build_gate


def _cross_validated_gate_scores(
    X: np.ndarray,
    y: np.ndarray,
    gate_name: str = "logistic_regression",
) -> tuple[Any, np.ndarray | None, dict[str, Any]]:
    """Return (fitted gate, OOF scores, meta) with robust fallbacks."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)

    if len(X) == 0:
        return None, None, {"status": "empty", "reason": "no examples"}

    unique_classes, class_counts = np.unique(y, return_counts=True)

    if len(unique_classes) < 2:
        return None, None, {
            "status": "fallback",
            "reason": f"only one class present: {unique_classes.tolist()}",
            "class_counts": {int(k): int(v) for k, v in zip(unique_classes, class_counts)},
        }

    min_count = int(class_counts.min())
    if min_count < 2:
        gate = build_gate(gate_name)
        gate.fit(X, y)
        scores = gate.predict_proba(X)[:, 1]
        return gate, scores, {
            "status": "train_only",
            "reason": "too few samples per class for cross-validation",
            "class_counts": {int(k): int(v) for k, v in zip(unique_classes, class_counts)},
        }

    cv = min(5, min_count)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    oof_scores = cross_val_predict(
        build_gate(gate_name),
        X,
        y,
        cv=splitter,
        method="predict_proba",
    )[:, 1]

    gate = build_gate(gate_name)
    gate.fit(X, y)

    return gate, oof_scores, {
        "status": "cross_validated",
        "cv_folds": cv,
        "gate_name": gate_name,
        "class_counts": {int(k): int(v) for k, v in zip(unique_classes, class_counts)},
    }


def _build_selective_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    total_count: int,
) -> dict[str, Any]:
    """Accuracy-coverage curve over actual unique score values."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    answered_count = len(scores)

    empty = {
        "thresholds": [],
        "coverages_total": [],
        "coverages_answered_only": [],
        "accuracies": [],
        "risks": [],
        "aurc": None,
        "risk_at_80_coverage": None,
        "risk_at_90_coverage": None,
        "coverage_at_5_risk": 0.0,
    }
    if answered_count == 0 or total_count == 0:
        return empty

    unique_scores = np.unique(scores)
    thresholds = np.concatenate([unique_scores[::-1], [unique_scores.min() - 1e-9]])

    seen_coverages: set[float] = set()
    actual_thresholds: list[float] = []
    coverages_total: list[float] = []
    coverages_answered: list[float] = []
    accuracies: list[float] = []

    for t in thresholds:
        mask = scores >= t
        kept = int(mask.sum())
        if kept == 0:
            continue
        cov_total = round(kept / total_count, 12)
        if cov_total in seen_coverages:
            continue
        seen_coverages.add(cov_total)
        actual_thresholds.append(float(t))
        coverages_total.append(cov_total)
        coverages_answered.append(kept / answered_count)
        accuracies.append(float(labels[mask].mean()))

    if not coverages_total:
        return empty

    risks = [1.0 - a for a in accuracies]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x compat
    aurc = float(trapezoid(risks, coverages_total)) if len(coverages_total) > 1 else None

    def _risk_at(target_cov: float) -> float:
        idx = int(np.argmin(np.abs(np.array(coverages_total) - target_cov)))
        return float(risks[idx])

    eligible = [c for c, r in zip(coverages_total, risks) if r <= 0.05]

    return {
        "thresholds": actual_thresholds,
        "coverages_total": coverages_total,
        "coverages_answered_only": coverages_answered,
        "accuracies": accuracies,
        "risks": risks,
        "aurc": aurc,
        "risk_at_80_coverage": _risk_at(0.8),
        "risk_at_90_coverage": _risk_at(0.9),
        "coverage_at_5_risk": float(max(eligible)) if eligible else 0.0,
    }


def compute_selective_curves(
    examples: list[dict[str, Any]],
    gate: Any = None,
    threshold_steps: int = 50,  # noqa: ARG001 — kept for backward-compat
    gate_name: str = "logistic_regression",
) -> dict[str, Any]:
    """Compute baseline (confidence-only) and proposed (gate) selective curves."""
    total = len(examples)
    answered = [ex for ex in examples if ex.get("category") != "abstain"]

    if total == 0 or not answered:
        return {
            "baseline": {},
            "proposed": {},
            "calibration": {},
            "gate_meta": {"status": "empty", "reason": "no answered examples"},
            "confidence_scores": [],
            "gate_scores": [],
            "labels": [],
        }

    X_answered, y_answered, answered_rows = prepare_features(examples, include_abstentions=False)
    confidence_scores = np.array([float(ex.get("confidence", 0.0)) for ex in answered_rows])

    _, cv_gate_scores, gate_meta = _cross_validated_gate_scores(
        X_answered, y_answered, gate_name=gate_name
    )
    if cv_gate_scores is None:
        gate_scores = confidence_scores.copy()
        gate_meta = {**gate_meta, "fallback_used": True, "fallback_scores": "confidence_only"}
    else:
        gate_scores = cv_gate_scores
        gate_meta = {**gate_meta, "fallback_used": False}

    baseline_curve = _build_selective_curve(confidence_scores, y_answered, total)
    proposed_curve = _build_selective_curve(gate_scores, y_answered, total)

    calibration = {
        "confidence_only": compute_calibration_metrics(confidence_scores, y_answered),
        "gate_score": compute_calibration_metrics(gate_scores, y_answered),
    }

    return {
        "baseline": baseline_curve,
        "proposed": proposed_curve,
        "calibration": calibration,
        "gate_meta": gate_meta,
        "num_answered_rows": len(answered_rows),
        "class_balance_answered": {
            "correct": int((y_answered == 1).sum()),
            "hallucinate": int((y_answered == 0).sum()),
        },
        "confidence_scores": confidence_scores.tolist(),
        "gate_scores": gate_scores.tolist(),
        "labels": y_answered.tolist(),
    }
