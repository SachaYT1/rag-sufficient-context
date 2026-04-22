"""Stratified selective curves by sufficiency and by difficulty level."""

from __future__ import annotations

from typing import Any

from src.gate.selective import compute_selective_curves


def stratified_selective_curves(
    examples: list[dict[str, Any]],
    strata_key: str = "sufficient",
    gate_name: str = "logistic_regression",
) -> dict[str, dict[str, Any]]:
    """Compute selective curves for each value of a stratification key.

    Defaults to ``sufficient`` which produces two curves: sufficient vs
    insufficient context. Empty strata are skipped.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        key = str(ex.get(strata_key))
        buckets.setdefault(key, []).append(ex)

    curves: dict[str, dict[str, Any]] = {}
    for key, subset in buckets.items():
        if len(subset) < 2:
            continue
        curves[key] = compute_selective_curves(subset, gate_name=gate_name)
    return curves
