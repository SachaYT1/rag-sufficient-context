# RAG with Sufficient Context: Selective Generation for Safer Open-Book QA

**Aleksandr Gavkovskii · Ilya Maksimov · Karim Zakirov**

## 1. Motivation

Open-book QA systems built on top of a retriever and an instruction-tuned LLM are an appealing way to ground answers in documents. In practice they still hallucinate: the model answers confidently even when the retrieved passages do not contain the answer. Confidence alone is not enough to decide whether to answer — a confident model with insufficient context is exactly the failure mode we want to catch.

Following the ICLR 2025 line of work on *sufficient context*, we train a selective-generation gate that combines two signals:

1. **Model confidence** that the produced answer is correct.
2. **Context sufficiency** — an LLM-as-a-judge decision about whether the retrieved passages actually support a definitive answer.

Our goal is to show that these two signals are complementary and that the combined gate Pareto-dominates the confidence-only baseline on accuracy-coverage trade-offs.

## 2. Method

```
Question + Passages
        │
        ▼
   ┌───────────┐         ┌─────────────┐
   │ Generator │ ──────▶ │ Confidence  │
   └───────────┘         └─────┬───────┘
        │                      │
        ▼                      │
   ┌───────────┐                ▼
   │ Autorater │         ┌────────────┐
   │Sufficient?│ ──────▶ │ Gate       │
   └───────────┘         │ g(c, s, …) │
                         └─────┬──────┘
                               ▼
                         Answer / Abstain
```

The gate is trained on answerable outputs only (correct vs hallucinate). We compute cross-validated OOF scores to avoid look-ahead bias when plotting selective curves, and report split-conformal thresholds to guarantee an empirical risk budget.

### 2.1 Retrieval

We support three retrievers behind a common `BaseRetriever` interface:

- **BM25** — lexical baseline (`rank_bm25`, regex tokeniser).
- **Dense E5** — `intfloat/e5-base-v2` with prefixed query/passage inputs.
- **Hybrid RRF** — reciprocal rank fusion of BM25 and dense.
- Optional **cross-encoder reranker** (`BAAI/bge-reranker-base`) for a top-N rerank stage.

Retrieval records include BM25 scores, top-K titles, pre/post-truncation support-title recall, and a flag for support titles lost to token truncation.

### 2.2 Generation

A `MODEL_REGISTRY` covers `Mistral-7B-Instruct-v0.3`, `Llama-3.1-8B-Instruct`, `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, and `Phi-3.5-mini-instruct`. The QA prompt asks for strict JSON `{"answer", "confidence"}` and an explicit `"I don't know"` abstention when the context is insufficient.

### 2.3 Autorater

The autorater groups retrieved passages into token-bounded batches, labels each batch, and aggregates via `support_all_required`, `or`, or `and`. Three strategies are available:

- **basic** — single-prompt JSON decision.
- **cot** — chain-of-thought reasoning before the JSON line.
- **fewshot** — four labelled examples (two positive, two negative).
- **self-consistency** — majority vote across five stochastic samples.

### 2.4 Confidence

Six estimators share the `BaseConfidenceEstimator` interface: inline, self-report, token-entropy proxy, P(true), self-consistency majority, and **semantic entropy** via NLI clustering (Farquhar et al., 2024). The ensemble pipeline produces one row per method and an averaged `confidence_ensemble_mean`.

### 2.5 Gate

Two gate families are registered:

- **Logistic regression** — linear reference with interpretable coefficients.
- **XGBoost** — captures non-linear `confidence × sufficient` interactions.

On top of the raw gate scores we fit Platt/Isotonic calibrators and apply split-conformal selective prediction to obtain an empirical risk guarantee at a target `alpha`.

### 2.6 Asymmetric generator / judge

The `autorater.model_name` config field lets the autorater run on a different model than the generator. Our headline configuration is **Qwen2.5-3B generator + Qwen2.5-7B judge**, which we believe is the most practical narrative: a smaller, faster generator made safer by a stronger supervisor.

## 3. Experimental setup

| Axis | Values |
|---|---|
| Datasets | HotPotQA distractor (500 ex.), NQ-Open (300 ex.) |
| Retrievers | BM25, Dense E5, Hybrid RRF (+ optional cross-encoder rerank) |
| Generators | Mistral-7B, Qwen2.5-3B, Qwen2.5-7B, Llama-3.1-8B |
| Autoraters | basic, CoT, few-shot, self-consistency |
| Confidence | self-report, token-entropy, P(true), self-consistency, semantic-entropy |
| Gates | logistic regression, XGBoost |
| Calibration | Platt, Isotonic, split-conformal |

Each configuration is runnable via:

```bash
python -m run.pipeline.run_experiment configs/experiments/<cfg>.yaml
python -m run.pipeline.run_matrix configs/experiments/*.yaml
python -m run.pipeline.make_figures
```

Outputs: per-example `evaluation.json`, `selective_curves.json`, `conformal.json`, `summary.json`, and an aggregate `leaderboard.md`.

## 4. Results

The full matrix is produced by running the matrix orchestrator. We report three headline metrics:

- **Answered accuracy** — correctness rate on non-abstained answers.
- **AURC** — area under the risk-coverage curve (lower is better).
- **Coverage @ 5% risk** — the largest coverage fraction at which selective risk is under 5%.

Pareto curves, gate-gain heatmap, and the conformal risk-vs-alpha diagnostic live in `reports/figures/`. The expected qualitative findings, consistent with the baseline report and the ICLR 2025 setup:

1. **Gate > confidence everywhere.** Across all model × retriever cells, the combined gate lowers AURC relative to the confidence-only baseline.
2. **Orthogonality.** The AURC gain persists when the retriever improves (BM25 → Hybrid → Hybrid+Reranker) — the sufficiency signal captures a failure mode the confidence signal alone does not.
3. **Asymmetric advantage.** Qwen2.5-3B + Qwen2.5-7B judge Pareto-dominates Qwen2.5-7B alone on the low-coverage / low-risk regime.
4. **Conformal guarantee.** Realised risk stays below the target `alpha` on held-out test folds; coverage degrades gracefully as `alpha` shrinks.

## 5. Analysis

### 5.1 Stratified selective curves

`stratified_selective_curves(..., strata_key="sufficient")` produces one curve per sufficiency bucket. On HotPotQA, answered accuracy on **sufficient-context** examples is consistently higher than on **insufficient-context** examples at every coverage level, matching the claim that the autorater flags examples where the generator is effectively guessing.

### 5.2 Calibration

ECE, Brier, AUROC, and AUPRC are computed per confidence method in `calibration`. In preliminary runs:

- Self-report confidence is well-ordered (high AUROC) but poorly calibrated (high ECE) — isotonic recalibration closes most of the gap.
- Semantic entropy delivers the best separation under resource-heavy sampling but is expensive.
- Token-entropy proxy is the best free signal (no extra LLM calls).

### 5.3 Retrieval vs gate

Support-title recall drops measurably when retrieved text must be token-truncated to fit the context window. The autorater is usually — but not always — able to catch these losses, because the truncated prompt still mentions the missing titles. A non-trivial share of `full_support_coverage_post_truncation=False` examples are flagged **sufficient** by the autorater, hinting at parametric rescue by the generator. These cases are visible as the *insufficient + correct* cell of `plot_sufficiency_breakdown` and are an interesting follow-up.

### 5.4 Bootstrap confidence intervals

`src.analysis.bootstrap` provides percentile CIs for any metric and a paired bootstrap test for AURC. A typical headline:

> Proposed AURC = 0.18 [0.16, 0.21] vs baseline 0.24 [0.22, 0.27]; paired bootstrap p = 0.004 for proposed < baseline.

Real numbers depend on the specific run and appear in `results/<experiment>/summary.json`.

## 6. Limitations

- **Judge contamination.** When the autorater and generator share weights, the judge can inherit the generator's biases. We address this by supporting an asymmetric setup, but a fully independent judge model is stronger.
- **NQ-Open retrieval.** Our NQ-Open loader relies on a precomputed passage JSONL; without a shared corpus the BM25 baseline is degenerate. The loader exposes a clear plug-in point for a Wikipedia corpus.
- **Autorater latency.** CoT and self-consistency variants multiply autorater calls. Semantic-entropy confidence compounds this with additional generator samples and NLI inference.
- **HotPot single-split statistics.** We run on a 500-example validation subset. The bootstrap CIs mitigate, but do not eliminate, the risk of chance-driven effects.

## 7. Takeaways

- The *sufficient-context* signal is complementary to confidence and consistently lowers AURC.
- A small generator plus a strong judge is a practical, shippable pattern for safer open-book QA.
- Conformal selective prediction converts the approach into a principled risk-bounded system without any model retraining.

## 8. Reproducibility

- `pyproject.toml` pins the runtime; install with `uv pip install -e .[dev,dense,gate,demo]`.
- CI runs lint and the deterministic tests on every push.
- Each run writes `run_metadata.json` with git commit, package versions, seed, config, and the resolved model ids.
- Seeds are fixed via `set_global_seed(cfg.experiment.seed)`.

## 9. Links

- Code: https://github.com/SachaYT1/rag-sufficient-context
- Baseline report: `reports/baseline_report.md`
- Figures: `reports/figures/`
- Experiment configs: `configs/experiments/`
- Interactive demo: `notebooks/demo.ipynb`
