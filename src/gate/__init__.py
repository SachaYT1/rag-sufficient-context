"""Selective-generation gate package.

Public surface is preserved for backward compatibility with existing notebooks
that imported from ``src.gate``.
"""

from __future__ import annotations

from src.gate.calibration import (
    compute_calibration_metrics,
    isotonic_calibrate,
    platt_calibrate,
)
from src.gate.conformal import conformal_selective, conformal_threshold
from src.gate.features import FEATURE_NAMES, prepare_features
from src.gate.models import GATE_REGISTRY, build_gate, register_gate
from src.gate.plots import (
    plot_accuracy_coverage,
    plot_calibration_curve,
    plot_gate_gain_heatmap,
    plot_score_distributions,
    plot_sufficiency_breakdown,
    plot_support_recall_vs_f1,
)
from src.gate.selective import compute_selective_curves


def train_gate(X, y, gate_name: str = "logistic_regression"):
    """Backward-compatible gate trainer (sklearn-style)."""
    gate = build_gate(gate_name)
    gate.fit(X, y)
    return gate


__all__ = [
    "FEATURE_NAMES",
    "GATE_REGISTRY",
    "build_gate",
    "compute_calibration_metrics",
    "compute_selective_curves",
    "conformal_selective",
    "conformal_threshold",
    "isotonic_calibrate",
    "platt_calibrate",
    "plot_accuracy_coverage",
    "plot_calibration_curve",
    "plot_gate_gain_heatmap",
    "plot_score_distributions",
    "plot_sufficiency_breakdown",
    "plot_support_recall_vs_f1",
    "prepare_features",
    "register_gate",
    "train_gate",
]
