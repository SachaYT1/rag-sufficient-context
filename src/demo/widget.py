"""Interactive ipywidgets demo for live threshold demonstrations."""

from __future__ import annotations

from typing import Any

import numpy as np


def selective_stats_at_threshold(
    scores: list[float] | np.ndarray,
    labels: list[float] | np.ndarray,
    total: int,
    threshold: float,
) -> dict[str, float]:
    """Compute coverage/risk/selective-accuracy at a given score threshold."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask = scores >= threshold
    kept = int(mask.sum())
    coverage = kept / max(1, total)
    if kept == 0:
        return {
            "threshold": float(threshold),
            "coverage_total": coverage,
            "selective_accuracy": float("nan"),
            "risk": float("nan"),
            "n_kept": 0,
        }
    acc = float(labels[mask].mean())
    return {
        "threshold": float(threshold),
        "coverage_total": coverage,
        "selective_accuracy": acc,
        "risk": 1.0 - acc,
        "n_kept": kept,
    }


def build_threshold_widget(
    scores: list[float] | np.ndarray,
    labels: list[float] | np.ndarray,
    total: int,
    step: float = 0.01,
) -> Any:
    """Return an ipywidgets.interactive that shows live metrics as you slide a threshold.

    Requires ``ipywidgets`` in the environment.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "ipywidgets is required for the interactive demo. Install with pip install ipywidgets."
        ) from exc

    slider = widgets.FloatSlider(
        value=0.5, min=0.0, max=1.0, step=step, description="threshold", continuous_update=False
    )
    output = widgets.Output()

    def _refresh(change: Any) -> None:
        with output:
            output.clear_output(wait=True)
            stats = selective_stats_at_threshold(scores, labels, total, slider.value)
            print(f"threshold         = {stats['threshold']:.3f}")
            print(f"coverage (total)  = {stats['coverage_total']:.3f}")
            print(f"selective accuracy= {stats['selective_accuracy']:.3f}")
            print(f"risk              = {stats['risk']:.3f}")
            print(f"# answered        = {stats['n_kept']}")

    slider.observe(_refresh, names="value")
    box = widgets.VBox([slider, output])
    display(box)
    _refresh(None)
    return box
