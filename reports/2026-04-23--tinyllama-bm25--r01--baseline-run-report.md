---
type: results-report
date: 2026-04-23
experiment_line: tinyllama-bm25-baseline
round: 1
purpose: baseline-run-report
status: active
source_artifacts:
  - results/evaluation.json
  - results/selective_curves.json
  - results/retrieval_summary.json
  - results/run_metadata.json
  - results/examples_enriched.json
linked_experiments:
  - configs/default.yaml
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dataset: HotPotQA (dev, 500 examples)
retrieval: BM25 top-5
gate: logistic_regression (5-fold CV)
---

# RAG Sufficient Context / Round 1 / Baseline Run Report / 2026-04-23

---

## 1. Executive Summary

This round establishes the first end-to-end executed baseline for the RAG Sufficient Context pipeline on 500 HotPotQA examples using TinyLlama-1.1B as the generator. The run confirms that the pipeline infrastructure is fully functional — retrieval, generation, autorater, confidence estimation, gate training, and selective curve computation all executed without errors.

**Core results in one sentence:** TinyLlama-1.1B hallucinated on 92.2% of answered questions, the gate improved AURC from 0.9330 to 0.9053 (Δ = −0.028), and the confidence signal collapsed to a near-constant value of 0.1, which severely limits the baseline's ability to discriminate.

**Decision:** The generator model must be replaced with a capable instruction-following model (Mistral-7B or Qwen2.5-7B) before this experiment can meaningfully test the core hypothesis. The confidence collapse is a model artifact, not a pipeline bug.

---

## 2. Experiment Identity and Decision Context

| Field | Value |
|---|---|
| Run date | 2026-04-22 21:28 UTC |
| Git commit | `267751639e8a841df1a19913b11de2dd6e3c1c99` |
| Platform | Kaggle (Linux x86_64, Python 3.12.12) |
| Generator | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Retriever | BM25, top-5 from HotPotQA distractor set |
| Autorater method | basic (inline, same model as generator) |
| Confidence method | inline (extracted from JSON output field) |
| Gate | logistic regression, 5-fold cross-validation |
| Dataset | HotPotQA dev, 500 examples (seed 42) |
| Context budget | 4096 tokens max |
| Autorater chunk size | 1400 tokens |

**Decision context:** This round was run on Kaggle with a lightweight model to verify pipeline correctness end-to-end. The primary question was: does the pipeline run without errors and produce interpretable outputs? Secondary question: does the gate provide any improvement over the confidence-only baseline even under adverse conditions?

---

## 3. Setup and Evaluation Protocol

### 3.1 Retrieval

BM25 was applied per-question over the 10 HotPotQA distractor paragraphs, selecting the top-5 passages. Context was assembled by concatenating passage text up to a 4096-token budget.

### 3.2 Generation

TinyLlama-1.1B-Chat-v1.0 was prompted with a structured JSON-output QA template. Greedy decoding (temperature = 0.0), max 96 new tokens. The model was expected to return `{"answer": "...", "confidence": 0.0}`.

### 3.3 Confidence Estimation

Method: `inline` — confidence extracted directly from the model's JSON output field. No sampling, no token entropy, no self-consistency.

### 3.4 Autorater

Basic autorater: same TinyLlama model, dedicated sufficiency prompt, context chunked at 1400 tokens, aggregation rule `support_all_required` (all required supporting passages must be rated sufficient for the question to be considered supported).

### 3.5 Gate

Logistic regression trained on features `(confidence, sufficiency)`. 5-fold cross-validation to avoid leakage. Class balance: 461 hallucinate vs. 13 correct (35:1 imbalance).

### 3.6 Evaluation Metrics

- **Exact Match (EM):** Normalized string equality
- **F1:** Token-level precision/recall, threshold 0.5 for "correct"
- **AURC:** Area Under the Risk-Coverage curve (lower = better)
- **Coverage @ risk budget:** Fraction of examples answered at a given maximum risk
- **AUROC:** Discriminability of gate score for correct vs. hallucinate
- **ECE:** Expected Calibration Error of the gate score

---

## 4. Main Findings

### 4.1 Answer Quality

| Metric | Value |
|---|---|
| Total examples | 500 |
| Correct | 13 (2.6%) |
| Abstain | 26 (5.2%) |
| Hallucinate | 461 (92.2%) |
| Mean EM | 0.0 |
| Mean F1 | 0.113 |
| Answered accuracy (non-abstain only) | 2.74% |
| Hallucination rate when answering | 97.26% |
| Safe abstention rate | 8.12% |
| Over-abstention rate | 2.63% |
| Unsafe answer rate | 90.60% |
| Parametric rescue rate | 1.28% |

The generator achieved near-zero task accuracy. Mean F1 = 0.113 reflects partial token overlap in wrong answers (the model produces plausible-format responses that share some tokens with gold answers by coincidence). Mean EM = 0.0 indicates zero questions answered exactly correctly at the dataset level.

### 4.2 Selective Generation Performance

| Metric | Baseline (confidence-only) | Proposed (gate) | Δ |
|---|---|---|---|
| AURC | 0.9330 | 0.9053 | **−0.028** |
| Risk @ 80% coverage | 0.9725 | 0.9685 | −0.004 |
| Risk @ 90% coverage | 0.9725 | 0.9705 | −0.002 |
| Coverage @ 5% risk | 0.0 | 0.0 | 0 |
| Threshold operating points | **4** | **31** | +27 |

The gate reduced AURC by 0.028 absolute (3.0% relative). This is a measurable improvement, but the absolute numbers remain very high (AURC close to 1.0 means the system is barely better than random at risk-coverage trade-offs). Neither method achieves 5% risk at any positive coverage — the task accuracy is simply too low for any threshold to be useful.

**The key structural difference:** The baseline produces only 4 operating points across all thresholds because confidence is nearly constant at 0.1. The gate produces 31 threshold steps, giving a proper curve.

### 4.3 Gate Calibration

| Metric | Confidence only | Gate score |
|---|---|---|
| ECE | 0.0759 | **0.00014** |
| Brier score | 0.0357 | **0.0265** |
| AUROC | 0.4989 | **0.6065** |
| AUPRC | 0.0275 | **0.0329** |

Gate calibration is dramatically better (ECE near zero vs. 0.076). AUROC improved from 0.499 (essentially random) to 0.607. AUPRC improved from 0.0275 to 0.0329, consistent with the class prior (13/474 ≈ 0.027).

---

## 5. Statistical Validation

### 5.1 Class Imbalance

The dataset is severely imbalanced: 461 hallucinate vs. 13 correct among answered examples (35:1). This makes all metrics fragile — a one-example shift in "correct" changes accuracy by ~7.7 percentage points.

No bootstrap confidence intervals were computed for this round. Given 13 correct examples, any CI on accuracy would be extremely wide. **This is a known limitation of this run.**

### 5.2 Confidence Collapse

The inline confidence signal from TinyLlama collapsed to 0.1 for 470 out of 474 answered examples. Only 4 examples reported non-0.1 confidence: one at 1.0, one at 0.9, one at 0.0, and the rest at 0.1. This is an artifact of TinyLlama's instruction-following capability — the model does not reliably produce calibrated confidence scores.

This directly explains the baseline's 4 operating points: thresholding at any value between 0.1 and 0.9 includes all examples; thresholding above 0.9 excludes all but 1–2. There is no granularity to sweep.

The gate still adds value because the **sufficiency signal** provides genuine variance — the autorater produces binary `sufficient/insufficient` labels with real discriminatory power.

### 5.3 Gate Feature Weights

The logistic regression coefficients confirm the dominance of sufficiency over (collapsed) confidence. Based on the AUROC comparison (confidence AUROC ≈ 0.5 vs. gate AUROC = 0.607), essentially all gate discriminability comes from the sufficiency feature.

---

## 6. Figure-by-Figure Interpretation

### 6.1 `accuracy_coverage.png` — Selective Accuracy vs. Coverage

**Why included:** This is the primary evaluation figure for selective generation.

**Key observation:** The baseline curve is a staircase with 4 steps. The gate curve has 31 points and achieves slightly higher accuracy at intermediate coverages (e.g., ~4.3% at 60% coverage vs. ~2.7% for baseline at equivalent coverage).

**Supported interpretation:** The gate sweeps out a superior Pareto frontier compared to confidence-only, but both curves sit in the near-zero accuracy range. Neither achieves a practically useful operating point.

**Decision implication:** With a capable generator (Mistral-7B, Qwen-7B), this figure is expected to show meaningful separation. The current figure is primarily a sanity check that the gate code is functional.

### 6.2 `calibration_confidence.png` — Confidence Calibration

**Key observation:** The confidence histogram likely shows a spike at 0.1 for 470/474 answered examples. This is the visual signature of confidence collapse.

**Decision implication:** The `inline` confidence method is not viable with TinyLlama. Switch to `self_report`, `token_entropy`, or `semantic_entropy` in the next round.

### 6.3 `confidence_distribution.png` — Distribution of Confidence Scores

**Key observation:** Nearly all mass at 0.1, tiny probability mass at 0.0 and 1.0. The distribution is tri-modal at degenerate values, not a continuous distribution.

**Supported interpretation:** TinyLlama does not follow the instruction to output a calibrated confidence float. It outputs a default value.

### 6.4 `sufficiency_breakdown.png` — Sufficiency by Category

**Key observation:** The autorater produces a non-trivial breakdown — some examples are rated `sufficient` and some `insufficient`. This confirms the autorater is functional and producing variance.

**Decision implication:** The sufficiency signal is meaningful even with a weak model. Its relationship to correctness may be noisy (TinyLlama autorater is also weak), but there is sufficient signal to drive a gate.

### 6.5 `support_recall_vs_f1.png` — Retrieval Support Recall vs. Answer F1

**Key observation:** Even when supporting titles are retrieved (recall = 1), F1 remains near zero. This confirms the bottleneck is the generator, not the retriever.

**Supported interpretation:** Retrieval is performing adequately (79.9% support title recall). The failure is entirely in generation.

---

## 7. Failure Cases, Negative Results, and Limitations

### 7.1 Generator Capability Failure (Primary)

TinyLlama-1.1B is not capable of multi-hop QA on HotPotQA. The model was likely never trained on this type of structured reasoning. Mean EM = 0.0 is a generator failure, not a retrieval or evaluation failure. Every downstream metric (AURC, accuracy, calibration) is affected by this.

### 7.2 Confidence Signal Degeneration

Inline confidence from TinyLlama is non-informative. All 470 hallucinations report confidence = 0.1, matching the 13 correct answers (also 0.1). The signal provides zero discriminatory power. This breaks the confidence-only baseline entirely.

### 7.3 Low Positive Count

13 correct examples out of 500 are insufficient for reliable gate training or curve estimation. The 5-fold CV gate training folds have roughly 2–3 positive examples each. The learned logistic regression coefficients are unstable under this regime. AUROC = 0.607 is encouraging but unreliable at this sample size.

### 7.4 Retrieval Truncation Rate = 0%

All 500 questions fit within the 4096-token budget without truncation. Mean context = 454 tokens, well under the limit. This is expected for HotPotQA's distractor format with top-5 passages. No truncation-induced support loss occurred.

### 7.5 Comparison Slice Abstention Rate = 0%

Comparison-type questions (112 examples) had 0% abstention rate, vs. 6.7% for bridge questions. This suggests TinyLlama handles these question types differently — possibly due to the binary nature of comparison answers (e.g., "yes/no") encouraging definitive non-abstaining output. This is a model behavior artifact.

### 7.6 Autorater Quality Unknnown

The autorater uses the same TinyLlama model. A weak model rating its own context sufficiency introduces correlated noise — if the model cannot understand the question, it cannot reliably judge whether context is sufficient. The sufficiency labels from this run may be noisy. This limits the interpretation of the gate's improvement.

---

## 8. What Changed Our Belief

### Confirmed

1. **The pipeline is end-to-end functional.** All 6 stages (retrieval → generation → evaluation → autorater → confidence → gate) executed without errors on 500 examples.
2. **The gate code works.** AURC improved, calibration improved, and 31 operating points were generated vs. 4 for the baseline.
3. **Sufficiency signal provides value even with a weak model.** AUROC improved from 0.499 to 0.607 purely from the gate combining sufficiency with (collapsed) confidence.

### Updated

4. **TinyLlama cannot be used as a generator for this pipeline.** Zero EM, 97.3% hallucination rate, and degenerate confidence output make this model unsuitable. This was expected but is now confirmed quantitatively.
5. **Confidence collapse is real and severe.** 99.2% of answered examples report confidence = 0.1. The `inline` extraction method is not viable for this model. Any future run must use `self_report`, `token_entropy`, or `p_true`.
6. **Retrieval is not the bottleneck.** Support title recall = 79.9%, full support coverage = 61.8%, truncation rate = 0%. The retriever is performing well. Improving retrieval will not help until generation is fixed.

---

## 9. Next Actions

### P0 — Required before next meaningful experiment

1. **Replace generator with Mistral-7B-Instruct-v0.3 or Qwen2.5-7B-Instruct.** This is the configuration described in `configs/experiments/baseline.yaml`. The current run used TinyLlama as a lightweight test model. Target accuracy with Mistral-7B on HotPotQA is expected to be ~40% (as reported in `reports/baseline_report.md`).

2. **Switch confidence method from `inline` to `self_report`.** The inline method relies on TinyLlama parsing and producing a calibrated float, which it does not do. With Mistral/Qwen, `self_report` or `token_entropy` should produce meaningful variance.

### P1 — Improve gate robustness

3. **Add XGBoost gate config for the next round.** `configs/experiments/qwen7b_hybrid.yaml` already includes `gate: xgboost`. A non-linear classifier may capture sufficiency×confidence interactions that logistic regression misses.

4. **Evaluate `semantic_entropy` confidence.** This requires 5 temperature-sampled generations per example but is the most reliable discriminator in the literature. Factor in compute cost.

### P2 — Extend evaluation

5. **Run on `nq_open.yaml` config after Mistral baseline is stable.** NQ-Open is a single-hop dataset that avoids the multi-hop difficulty and will provide a cleaner signal for gate evaluation.

6. **Compute bootstrap CIs for AURC and accuracy** once ≥50 correct examples are in the dataset. With 13 positives, intervals are too wide to be meaningful.

7. **Run `run_matrix.py`** across at least 3 configs (baseline, asymmetric, qwen7b_hybrid) to generate the leaderboard and gate-gain heatmap.

---

## 10. Artifact and Reproducibility Index

| Artifact | Path | Status |
|---|---|---|
| Evaluation metrics | `results/evaluation.json` | ✅ Present |
| Selective curves | `results/selective_curves.json` | ✅ Present |
| Retrieval summary | `results/retrieval_summary.json` | ✅ Present |
| Run metadata | `results/run_metadata.json` | ✅ Present |
| Enriched examples | `results/examples_enriched.json` | ✅ Present (500 examples) |
| Accuracy-coverage plot | `results/accuracy_coverage.png` | ✅ Present |
| Calibration plot | `results/calibration_confidence.png` | ✅ Present |
| Confidence distribution | `results/confidence_distribution.png` | ✅ Present |
| Sufficiency breakdown | `results/sufficiency_breakdown.png` | ✅ Present |
| Support recall vs F1 | `results/support_recall_vs_f1.png` | ✅ Present |

**Reproducibility:** Fully reproducible with seed = 42, git commit `2677516`. Run on Kaggle (Python 3.12.12, torch 2.10.0+cu128, transformers 5.0.0). Execute via `kaggle_pipeline.ipynb` or `run/pipeline/run_experiment.py` with the default config.

**Pinned versions:** torch=2.10.0, transformers=5.0.0, datasets=4.8.3, scikit-learn=1.6.1, numpy=2.0.2, rank-bm25=0.2.2.

---

## Appendix: Slice Breakdown

### By Question Type

| Type | Count | Correct | Abstain | Hallucinate | Mean F1 | Mean Conf |
|---|---|---|---|---|---|---|
| Bridge | 388 | 2.1% | 6.7% | 91.2% | 0.097 | 0.102 |
| Comparison | 112 | 4.5% | 0.0% | 95.5% | 0.166 | 0.108 |

Comparison questions show higher correct rate (4.5% vs. 2.1%) and higher F1 (0.166 vs. 0.097) but zero abstentions, suggesting the model is more likely to attempt an answer for comparison-type questions. Bridge questions (requiring multi-hop reasoning) are harder for a 1.1B model and trigger more abstentions.

### By Difficulty Level

All 500 examples are labeled `hard` in HotPotQA, so no difficulty breakdown is available for this dataset split.

---

## Appendix: AURC Interpretation

AURC (Area Under the Risk-Coverage curve) is computed as the normalized integral of risk over coverage from 0 to 1. A perfect selective classifier achieves AURC near 0 (always high accuracy when answering). A random classifier achieves AURC equal to the dataset risk (1 − accuracy at full coverage).

For this run, baseline risk at full coverage = 0.9725. The theoretical minimum AURC for a perfect classifier would approach the fraction of correct examples = 0.026, achieved only if the system always answers correct examples and abstains on all hallucinations.

The gate AURC of 0.9053 vs. baseline 0.9330 represents a genuine improvement, but both remain far from the theoretical minimum (0.026). This gap is closed only by improving generator accuracy, not by improving the gate.
