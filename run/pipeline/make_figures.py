"""Generate headline figures from a matrix of experiment results.

Reads ``results/<experiment>/selective_curves.json`` and ``conformal.json``
for each experiment listed in ``results/matrix/summary.json`` and renders:

- ``reports/figures/pareto_selective.pdf`` — accuracy-coverage curves
  overlaid across experiments.
- ``reports/figures/gate_gain_heatmap.pdf`` — model x retriever AURC gain.
- ``reports/figures/conformal_guarantee.pdf`` — realised risk vs target
  alpha per experiment.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.gate.plots import plot_gate_gain_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_pareto(summaries: list[dict[str, Any]], results_root: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for summary in summaries:
        name = summary.get("experiment")
        curve_path = results_root / name / "selective_curves.json"
        if not curve_path.exists():
            continue
        curves = _load_json(curve_path)
        proposed = curves.get("proposed", {})
        if proposed.get("coverages_total"):
            ax.plot(
                proposed["coverages_total"],
                proposed["accuracies"],
                marker="o",
                markersize=3,
                label=name,
            )
    ax.set_xlabel("End-to-end coverage")
    ax.set_ylabel("Selective accuracy")
    ax.set_title("Selective accuracy vs coverage — all experiments")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(summaries: list[dict[str, Any]], out_path: Path) -> None:
    matrix: dict[tuple[str, str], float] = {}
    for summary in summaries:
        name = summary.get("experiment", "")
        b = summary.get("baseline_aurc")
        p = summary.get("proposed_aurc")
        if b is None or p is None:
            continue
        parts = name.split("_")
        model = parts[0] if parts else name
        retriever = parts[1] if len(parts) > 1 else "unknown"
        matrix[(model, retriever)] = float(b - p)  # positive = gate wins
    if matrix:
        plot_gate_gain_heatmap(matrix, save_path=str(out_path))


def _plot_conformal(summaries: list[dict[str, Any]], results_root: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    alphas: list[float] = []
    risks: list[float] = []
    labels: list[str] = []
    for summary in summaries:
        name = summary.get("experiment", "")
        cpath = results_root / name / "conformal.json"
        if not cpath.exists():
            continue
        data = _load_json(cpath)
        alpha = data.get("alpha")
        risk = data.get("risk")
        if alpha is None or risk is None:
            continue
        alphas.append(float(alpha))
        risks.append(float(risk))
        labels.append(name)

    if not alphas:
        logger.info("No conformal outputs available — skipping plot")
        plt.close(fig)
        return

    ax.scatter(alphas, risks)
    for a, r, lab in zip(alphas, risks, labels):
        ax.annotate(lab, (a, r), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="target = realised")
    ax.set_xlabel("Target risk alpha")
    ax.set_ylabel("Realised selective risk")
    ax.set_title("Conformal risk guarantee")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate headline figures.")
    parser.add_argument("--matrix", default="results/matrix/summary.json")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", default="reports/figures")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir)

    if not matrix_path.exists():
        logger.error("Matrix summary not found: %s", matrix_path)
        return 1

    summaries = _load_json(matrix_path)
    _plot_pareto(summaries, results_root, out_dir / "pareto_selective.pdf")
    _plot_heatmap(summaries, out_dir / "gate_gain_heatmap.pdf")
    _plot_conformal(summaries, results_root, out_dir / "conformal_guarantee.pdf")
    logger.info("Wrote figures to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
