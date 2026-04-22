"""Selective-generation gate, calibration metrics and plotting utilities."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def prepare_features(
    examples: list[dict[str, Any]],
    include_abstentions: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extract model 
    By default the gate is trained on answerable outputs only:
    1 = correct, 0 = hallucinate.
    """
    rows = [ex for ex in examples if include_abstentions or ex.get("category") != "abstain"]
    X, y = [], []
    for ex in rows:
        confidence = float(ex.get("confidence", 0.0))
        sufficient = float(bool(ex.get("sufficient", False)))
        positive_chunk_ratio = float(ex.get("positive_chunk_ratio", 0.0))
        support_recall_post = ex.get("support_title_recall_post_truncation")
        support_recall_post = float(support_recall_post) if support_recall_post is not None else 0.0
        X.append([confidence, sufficient, positive_chunk_ratio, support_recall_post])
        y.append(float(ex.get("category") == "correct"))
    return np.array(X), np.array(y), rows


def train_gate(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Train logistic regression gate on all provided examples."""
    gate = LogisticRegression(random_state=42, max_iter=1000)
    gate.fit(X, y)
    return gate


def _cross_validated_gate_scores(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[LogisticRegression | None, np.ndarray | None, dict[str, Any]]:
    """Return (fitted gate, OOF scores, meta) with robust fallback.

    Falls back to None scores when fewer than 2 classes are present.
    Falls back to train-only scores when classes are too rare for cross-val.
    """
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
        gate = LogisticRegression(random_state=42, max_iter=1000)
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
        LogisticRegression(random_state=42, max_iter=1000),
        X,
        y,
        cv=splitter,
        method="predict_proba",
    )[:, 1]

    gate = LogisticRegression(random_state=42, max_iter=1000)
    gate.fit(X, y)

    return gate, oof_scores, {
        "status": "cross_validated",
        "cv_folds": cv,
        "class_counts": {int(k): int(v) for k, v in zip(unique_classes, class_counts)},
    }


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""
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
    """Compute calibration and ranking metrics."""
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


def _build_selective_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    total_count: int,
) -> dict[str, Any]:
    """Build a selective accuracy-coverage curve using actual score values as thresholds.

    Using the actual unique score values avoids empty threshold bins and the
    artificial accuracy=1.0 spike that appears at coverage=0 with linspace grids.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    answered_count = len(scores)

    if answered_count == 0 or total_count == 0:
        return {
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

    unique_scores = np.unique(scores)
    # Sweep thresholds from high (only best examples) to low (all examples pass)
    thresholds = np.concatenate([
        unique_scores[::-1],
        [unique_scores.min() - 1e-9],  # ensure at least the all-included point
    ])

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
        return {
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

    risks = [1.0 - a for a in accuracies]
    aurc = float(np.trapz(risks, coverages_total)) if len(coverages_total) > 1 else None

    def _risk_at(target_cov: float) -> float | None:
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
    gate: LogisticRegression | None = None,
    threshold_steps: int = 50,
) -> dict[str, Any]:
    """Compute selective accuracy-coverage curves with calibration metrics.

    Always trains an internal cross-validated gate from example features.
    The ``gate`` parameter is accepted for backward compatibility but not used
    (cross-validation provides unbiased OOF scores for plotting).
    ``threshold_steps`` is retained for API compatibility.
    """
    total = len(examples)
    answered_examples = [ex for ex in examples if ex.get("category") != "abstain"]

    if total == 0 or not answered_examples:
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

    _, cv_gate_scores, gate_meta = _cross_validated_gate_scores(X_answered, y_answered)
    if cv_gate_scores is None:
        # Single class or no data: proposed curve falls back to confidence
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


def plot_accuracy_coverage(
    curves: dict[str, Any],
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot end-to-end selective accuracy versus total coverage."""
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
    """Plot category distribution by sufficiency."""
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
    """Plot a reliability diagram (skips empty bins)."""
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=float)

    if len(scores) == 0:
        print(f"Skipping calibration plot '{title}': no data.")
        return

    if len(np.unique(labels)) < 2:
        print(f"Skipping calibration plot '{title}': only one class in labels.")
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
    """Plot score histograms split by response category."""
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
    """Scatter plot of support title recall (post-truncation) vs F1, coloured by category."""
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