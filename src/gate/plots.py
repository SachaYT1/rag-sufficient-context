"""Matplotlib plotting utilities for gate and sufficiency diagnostics."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_coverage(
    curves: dict[str, Any],
    save_path: str | None = None,
    show: bool = False,
) -> None:
    baseline = curves.get("baseline", {})
    proposed = curves.get("proposed", {})

    if not baseline.get("coverages_total") or not proposed.get("coverages_total"):
        print("Skipping accuracy-coverage plot: insufficient curve data.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        baseline["coverages_total"],
        baseline["accuracies"],
        marker="o",
        markersize=3,
        label="Baseline (confidence only)",
    )
    ax.plot(
        proposed["coverages_total"],
        proposed["accuracies"],
        marker="s",
        markersize=3,
        label="Proposed (gate score)",
    )
    ax.set_xlabel("End-to-end coverage")
    ax.set_ylabel("Selective accuracy")
    ax.set_title("Selective Accuracy vs Coverage")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_sufficiency_breakdown(
    examples: list[dict[str, Any]],
    save_path: str | None = None,
    show: bool = False,
) -> None:
    groups = {
        "Sufficient": {"correct": 0, "abstain": 0, "hallucinate": 0},
        "Insufficient": {"correct": 0, "abstain": 0, "hallucinate": 0},
    }
    for ex in examples:
        group = "Sufficient" if ex.get("sufficient", False) else "Insufficient"
        category = ex.get("category", "hallucinate")
        if category in groups[group]:
            groups[group][category] += 1

    categories = ["correct", "abstain", "hallucinate"]
    x = np.arange(len(categories))
    width = 0.35

    sufficient_total = max(1, sum(groups["Sufficient"].values()))
    insufficient_total = max(1, sum(groups["Insufficient"].values()))
    sufficient_pct = [groups["Sufficient"][c] / sufficient_total * 100 for c in categories]
    insufficient_pct = [groups["Insufficient"][c] / insufficient_total * 100 for c in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, sufficient_pct, width, label=f"Sufficient (n={sufficient_total})")
    ax.bar(x + width / 2, insufficient_pct, width, label=f"Insufficient (n={insufficient_total})")
    ax.set_ylabel("Percentage")
    ax.set_title("Response Categories by Context Sufficiency")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_calibration_curve(
    scores: list[float] | np.ndarray,
    labels: list[float] | np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=float)

    if len(scores) == 0 or len(np.unique(labels)) < 2:
        print(f"Skipping calibration plot '{title}': insufficient data.")
        return

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores <= hi) if i == n_bins - 1 else (scores >= lo) & (scores < hi)
        if not mask.any():
            continue
        xs.append(float(scores[mask].mean()))
        ys.append(float(labels[mask].mean()))

    if len(xs) < 2:
        print(f"Skipping calibration plot '{title}': fewer than 2 non-empty bins.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(xs, ys, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_score_distributions(
    examples: list[dict[str, Any]],
    score_key: str = "confidence",
    title: str | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    categories = ["correct", "hallucinate", "abstain"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for category in categories:
        values = [float(ex.get(score_key, 0.0)) for ex in examples if ex.get("category") == category]
        if values:
            ax.hist(values, bins=20, alpha=0.5, label=category.capitalize())

    ax.set_xlabel(score_key)
    ax.set_ylabel("Count")
    ax.set_title(title or f"{score_key} distribution by category")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_support_recall_vs_f1(
    per_example: list[dict[str, Any]],
    save_path: str | None = None,
    show: bool = False,
) -> None:
    category_colors = {"correct": "green", "hallucinate": "red", "abstain": "gray"}
    groups: dict[str, tuple[list[float], list[float]]] = {c: ([], []) for c in category_colors}

    for row in per_example:
        recall = row.get("support_title_recall_post_truncation")
        if recall is None:
            continue
        cat = str(row.get("category", "unknown"))
        if cat in groups:
            groups[cat][0].append(float(recall))
            groups[cat][1].append(float(row.get("f1", 0.0)))

    total_points = sum(len(v[0]) for v in groups.values())
    if total_points == 0:
        print("Skipping support_recall_vs_f1 plot: no data with recall values.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for cat, (xs, ys) in groups.items():
        if xs:
            ax.scatter(xs, ys, alpha=0.5, color=category_colors[cat], label=cat.capitalize(), s=20)
    ax.set_xlabel("Support title recall (post-truncation)")
    ax.set_ylabel("F1 score")
    ax.set_title("Support Recall vs F1")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_gate_gain_heatmap(
    matrix: dict[tuple[str, str], float],
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
    save_path: str | None = None,
    show: bool = False,
    title: str = "AURC gain of gate over confidence baseline",
) -> None:
    """Render a model x retriever heatmap of AURC gains."""
    if not matrix:
        print("Skipping gain heatmap: empty matrix.")
        return
    rows = row_order or sorted({r for r, _ in matrix})
    cols = col_order or sorted({c for _, c in matrix})
    data = np.array([[matrix.get((r, c), np.nan) for c in cols] for r in rows])

    fig, ax = plt.subplots(figsize=(max(4, len(cols) * 1.2), max(3, len(rows) * 0.8)))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-abs(np.nanmax(np.abs(data))), vmax=abs(np.nanmax(np.abs(data))))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
