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
  retrieval.py      # BM25 retriever + context construction
  generation.py     # LLaMA-3.1-8B prompting and answer extraction
  evaluation.py     # EM/F1 metrics, abstain detection, categorization
  autorater.py      # Sufficient context binary classifier
  confidence.py     # P(Correct) confidence estimation
  gate.py           # Logistic regression gate + visualization
  utils.py          # Shared helpers
configs/
  default.yaml      # Hyperparameters
notebooks/
  main_pipeline.ipynb  # End-to-end Colab notebook
reports/
  baseline_report.md   # Baseline report
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
