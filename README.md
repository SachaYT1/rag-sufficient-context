# RAG Sufficient Context: Selective Generation Pipeline

Explaining and Reducing Hallucinations in RAG via "Sufficient Context" — Selective Generation inspired by ICLR 2025.

**Authors:** Aleksandr Gavkovskii, Ilya Maksimov, Karim Zakirov

## Overview

This project builds a RAG evaluation and intervention pipeline for open-book QA that demonstrates a **sufficient-context signal**, combined with **model confidence**, improves the accuracy-coverage trade-off by making better answer vs. abstain decisions.

Based on: [Sufficient Context: A New Lens on Systems Retrieval Augmented Generation](https://openreview.net/forum?id=sufficient-context) (ICLR 2025)

## Key Idea

```
Question + Retrieved Context
        |
        v
   +-----------+        +------------+
   | Generator | -----> | Confidence |
   | (LLaMA-3) |        | P(Correct) |
   +-----------+        +-----+------+
        |                     |
        v                     v
   +-----------+     +------------------+
   | Autorater | --> | Gate (LogReg)    |
   | Sufficient|     | g(conf, suffic.) |
   +-----------+     +--------+---------+
                              |
                     Answer or Abstain
```

## Quick Start

### Run on Google Colab (recommended)

Open `notebooks/main_pipeline.ipynb` in Google Colab with GPU runtime.

### Local Setup

```bash
git clone https://github.com/SachaYT1/rag-sufficient-context.git
cd rag-sufficient-context
pip install -r requirements.txt
```

## Project Structure

```
src/
  config.py              # Typed, frozen PipelineConfig dataclasses
  prompts.py             # Central prompt loader
  utils.py               # Tokenisation, generation, metadata
  evaluation.py          # EM/F1, abstain detection, categorisation
  data/                  # Dataset loaders (HotPotQA, NQ-Open)
  retrieval/             # BM25, Dense E5, Hybrid RRF, Cross-encoder + registry
  generation/            # QA prompt, HF loader, model registry
  autorater/             # basic/CoT/fewshot/self-consistency + aggregation
  confidence/            # inline, self-report, token-entropy, p-true,
                         # self-consistency, semantic-entropy, ensemble
  gate/                  # features, LogReg/XGB gates, selective curves,
                         # calibration, split-conformal, plots
  analysis/              # bootstrap CIs, paired test, stratified curves
  demo/                  # ipywidgets threshold widget

configs/
  default.yaml                 # Shared defaults
  experiments/                 # One YAML per named experiment
prompts/                       # Prompt templates stored as files
run/pipeline/
  run_experiment.py            # Single-config runner
  run_matrix.py                # Aggregate leaderboard across configs
  make_figures.py              # Headline figures from the matrix
notebooks/
  main_pipeline.ipynb          # End-to-end Colab notebook
  demo.ipynb                   # Live-threshold demo
reports/
  baseline_report.md
  final_report.md              # Final paper-style report
  presentation_outline.md      # Slide plan
  figures/                     # Generated figures
tests/                         # pytest smoke suite
.github/workflows/ci.yml       # Ruff + pytest on push
pyproject.toml                 # uv/pip metadata and optional extras
```

## Running an experiment

```bash
uv pip install -e .[dev,dense,gate,demo]

# Single config
python -m run.pipeline.run_experiment configs/experiments/qwen7b_hybrid.yaml

# Full matrix + leaderboard
python -m run.pipeline.run_matrix configs/experiments/*.yaml

# Headline figures
python -m run.pipeline.make_figures
```

## Pipeline

1. **Dataset**: HotPotQA dev subset (500 examples)
2. **Retrieval**: BM25 over distractor passages, top-5, context budget 4096 tokens
3. **Generation**: LLaMA-3.1-8B-Instruct, greedy decoding, JSON output with answer + confidence
4. **Evaluation**: EM/F1, abstention detection, categorization (correct/abstain/hallucinate)
5. **Autorater**: Sufficient context classification via LLM with chunking + OR-aggregation
6. **Gate**: Logistic regression on (confidence, sufficiency) with threshold sweep
7. **Comparison**: Selective accuracy-coverage curves (baseline vs. proposed)

## Results

Expected outcomes (consistent with ICLR 2025 paper):
- Combining sufficient-context with confidence improves selective accuracy at fixed coverage
- Higher hallucination rate when context is insufficient
- The sufficiency signal provides an orthogonal dimension beyond confidence alone

## License

MIT
