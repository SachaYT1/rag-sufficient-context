"""Batch confidence estimation over a list of examples."""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.confidence.registry import build_confidence_estimator


def estimate_confidence_batch(
    examples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    use_inline: bool = True,
    method: str | None = None,
    method_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    resolved_method = method or ("inline" if use_inline else "self_report")
    estimator = build_confidence_estimator(resolved_method, method_config)

    results: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc=f"Estimating confidence ({resolved_method})"):
        confidence, diagnostics = estimator.estimate(ex, model, tokenizer)
        results.append(
            {
                **ex,
                "confidence": float(confidence),
                "confidence_method": diagnostics.get("confidence_method", resolved_method),
                "confidence_diagnostics": diagnostics,
            }
        )
    return results


def estimate_confidence_ensemble(
    examples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    methods: list[str],
    method_configs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run multiple confidence methods and attach each score under
    ``confidence_<method>`` without overwriting the primary ``confidence``.
    """
    method_configs = method_configs or {}
    estimators = {m: build_confidence_estimator(m, method_configs.get(m, {})) for m in methods}

    enriched: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc=f"Confidence ensemble ({','.join(methods)})"):
        out = dict(ex)
        scores: dict[str, float] = {}
        for m, estimator in estimators.items():
            conf, diag = estimator.estimate(out, model, tokenizer)
            scores[m] = float(conf)
            out[f"confidence_{m}"] = float(conf)
            out.setdefault("confidence_diagnostics_by_method", {})[m] = diag
        # Average as a simple ensemble score
        out["confidence_ensemble_mean"] = sum(scores.values()) / max(1, len(scores))
        enriched.append(out)
    return enriched
