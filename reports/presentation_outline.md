# Presentation Outline — RAG with Sufficient Context

Target duration: 10–12 minutes, ~12 slides.

## Slide 1 — Title
- Project name, authors, affiliation.
- One-liner: *"A retriever that knows when not to answer."*

## Slide 2 — The problem
- Confidence-only RAG hallucinates when passages look plausible but miss the key fact.
- Example pair: one visibly sufficient prompt, one visibly insufficient prompt with the same confidence score.

## Slide 3 — Idea
- Two complementary signals: **confidence** and **sufficient context**.
- Gate combines them into a single answer/abstain decision.
- Diagram from `final_report.md` §2.

## Slide 4 — Pipeline
- Dataset → retriever → generator → confidence estimator → autorater → gate → selective output.
- Show the directory layout (`src/{retrieval,generation,autorater,confidence,gate,analysis}`) to emphasise the modular design.

## Slide 5 — Models & retrievers
- Generator registry: Qwen2.5-3B / 7B, Llama-3.1-8B, Mistral-7B, Phi-3.5-mini.
- Retriever registry: BM25, Dense E5, Hybrid RRF, Cross-encoder reranker.
- Autorater strategies: basic, CoT, few-shot, self-consistency.
- Confidence estimators: self-report, P(true), token-entropy, self-consistency, semantic entropy.

## Slide 6 — Main result
- Accuracy-coverage Pareto curve: baseline (confidence only) vs proposed (gate).
- One number on the slide: `Coverage @ 5% risk` delta.

## Slide 7 — Orthogonality
- Gate-gain heatmap (`reports/figures/gate_gain_heatmap.pdf`): model × retriever.
- Message: the sufficiency signal is ortogonal to retrieval quality and to model size.

## Slide 8 — Asymmetric setup
- Pareto plot for three configurations: small-generator alone, large-generator alone, small-generator + large judge.
- Key line: *"A small generator supervised by a large judge beats a large generator running alone at low-risk operating points."*

## Slide 9 — Conformal guarantee
- Scatter of `alpha` vs realised risk with `y = x` reference.
- Call out the risk budget: *given an alpha, the selective risk stays below alpha on held-out data.*

## Slide 10 — Statistical significance
- Bootstrap CIs for AURC per experiment and paired bootstrap test vs baseline.
- One sentence: *"Proposed AURC is lower than baseline with p < 0.01."*

## Slide 11 — Live demo
- Open `notebooks/demo.ipynb`.
- Slide a threshold; show coverage, selective accuracy, and risk update in real time.
- Switch between `confidence` and `gate_score` threshold views.

## Slide 12 — Takeaways and next steps
- Sufficient-context is a practical, cheap, complementary signal.
- Asymmetric generator/judge is a deployment-ready pattern.
- Next: open-ended datasets (PopQA popularity strata), a semantic-entropy-calibrated gate, and a learned abstention head.

## Backup slides
- Full ablation table per config (`results/matrix/leaderboard.md`).
- Calibration reliability diagrams for all confidence methods.
- Failure-mode examples: insufficient-correct (parametric rescue) and sufficient-hallucinate (retriever surfaced the right title but generator ignored it).
