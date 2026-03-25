"""Selective generation gate: logistic regression on (confidence, sufficiency)."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict


def prepare_features(examples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from evaluated examples.

    Features: [confidence, sufficient (0/1)]
    Labels: 1 = correct, 0 = error (hallucinate)
    Abstentions are excluded from training.
    """
    X, y = [], []
    for ex in examples:
        if ex.get("category") == "abstain":
            continue
        confidence = float(ex.get("confidence", 0.0))
        sufficient = float(ex.get("sufficient", False))
        correct = float(ex.get("category") == "correct")
        X.append([confidence, sufficient])
        y.append(correct)
    return np.array(X), np.array(y)


def train_gate(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Train logistic regression gate."""
    gate = LogisticRegression(random_state=42, max_iter=1000)
    gate.fit(X, y)
    return gate


def compute_selective_curves(
    examples: list[dict],
    gate: LogisticRegression | None = None,
    threshold_steps: int = 50,
) -> dict:
    """Compute selective accuracy vs. coverage curves.

    Returns dict with:
      - 'proposed': curves using gate (confidence + sufficiency)
      - 'baseline': curves using confidence only
    """
    # Filter out abstentions for curve computation
    non_abstain = [ex for ex in examples if ex.get("category") != "abstain"]
    if not non_abstain:
        return {"proposed": {}, "baseline": {}}

    confidences = np.array([float(ex.get("confidence", 0.0)) for ex in non_abstain])
    sufficients = np.array([float(ex.get("sufficient", False)) for ex in non_abstain])
    corrects = np.array([float(ex.get("category") == "correct") for ex in non_abstain])

    thresholds = np.linspace(0, 1, threshold_steps)

    # Baseline: confidence-only
    baseline_coverages = []
    baseline_accuracies = []
    for t in thresholds:
        mask = confidences >= t
        coverage = mask.sum() / len(non_abstain)
        accuracy = corrects[mask].mean() if mask.sum() > 0 else 0.0
        baseline_coverages.append(coverage)
        baseline_accuracies.append(accuracy)

    # Proposed: gate score (confidence + sufficiency)
    # Use cross-validated predictions to avoid train/test contamination
    proposed_coverages = []
    proposed_accuracies = []
    if gate is not None:
        X = np.column_stack([confidences, sufficients])
        # Cross-validated probability estimates for unbiased evaluation
        try:
            gate_scores = cross_val_predict(
                LogisticRegression(random_state=42, max_iter=1000),
                X, corrects, cv=5, method="predict_proba",
            )[:, 1]
        except ValueError:
            # Fallback if CV fails (e.g., too few samples)
            gate_scores = gate.predict_proba(X)[:, 1]

        for t in thresholds:
            mask = gate_scores >= t
            coverage = mask.sum() / len(non_abstain)
            accuracy = corrects[mask].mean() if mask.sum() > 0 else 0.0
            proposed_coverages.append(coverage)
            proposed_accuracies.append(accuracy)
    else:
        proposed_coverages = baseline_coverages
        proposed_accuracies = baseline_accuracies

    return {
        "baseline": {
            "thresholds": thresholds.tolist(),
            "coverages": baseline_coverages,
            "accuracies": baseline_accuracies,
        },
        "proposed": {
            "thresholds": thresholds.tolist(),
            "coverages": proposed_coverages,
            "accuracies": proposed_accuracies,
        },
    }


def plot_accuracy_coverage(curves: dict, save_path: str | None = None) -> None:
    """Plot selective accuracy vs. coverage curves."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(
        curves["baseline"]["coverages"],
        curves["baseline"]["accuracies"],
        "b-o", markersize=3, label="Baseline (confidence only)",
    )
    ax.plot(
        curves["proposed"]["coverages"],
        curves["proposed"]["accuracies"],
        "r-s", markersize=3, label="Proposed (confidence + sufficiency)",
    )

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Selective Accuracy", fontsize=12)
    ax.set_title("Selective Accuracy vs. Coverage", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sufficiency_breakdown(examples: list[dict], save_path: str | None = None) -> None:
    """Plot correct/abstain/hallucinate breakdown by sufficiency."""
    sns.set_theme(style="whitegrid")

    sufficient = {"correct": 0, "abstain": 0, "hallucinate": 0}
    insufficient = {"correct": 0, "abstain": 0, "hallucinate": 0}

    for ex in examples:
        cat = ex.get("category", "hallucinate")
        if ex.get("sufficient", False):
            sufficient[cat] += 1
        else:
            insufficient[cat] += 1

    categories = ["correct", "abstain", "hallucinate"]
    suf_vals = [sufficient[c] for c in categories]
    insuf_vals = [insufficient[c] for c in categories]

    # Normalize to percentages
    suf_total = sum(suf_vals) or 1
    insuf_total = sum(insuf_vals) or 1
    suf_pct = [v / suf_total * 100 for v in suf_vals]
    insuf_pct = [v / insuf_total * 100 for v in insuf_vals]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, suf_pct, width, label=f"Sufficient (n={suf_total})", color="steelblue")
    ax.bar(x + width / 2, insuf_pct, width, label=f"Insufficient (n={insuf_total})", color="salmon")

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Response Categories by Context Sufficiency", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)
    ax.legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
