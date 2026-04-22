# RAG Sufficient Context — Project Guide

Selective-generation RAG pipeline on HotPotQA. Detects **sufficient context** with an autorater and combines it with **P(Correct)** confidence to decide whether to answer or abstain. Based on *Sufficient Context: A New Lens on RAG* (ICLR 2025).

**Authors**: Aleksandr Gavkovskii, Ilya Maksimov, Karim Zakirov
**Current branch**: `karim` (main branch: `main`)

---

## Stack

- **Python** managed via `uv` (preferred) or `pip install -r requirements.txt`
- **Model**: Mistral-7B-Instruct-v0.3 (migrated from LLaMA-3.1-8B to avoid HF gating — see commit `f36e25c`)
- **Core deps**: `transformers==4.45.2`, `torch==2.4.1`, `accelerate`, `datasets==2.21.0`, `rank-bm25`, `scikit-learn`, `PyYAML`
- **Config**: YAML files under `configs/` (Hydra not yet wired in — plain `yaml.safe_load` via `src/config.py`)
- **Entry points**: `notebooks/main_pipeline.ipynb` (Colab GPU) and `run/pipeline/`

---

## Layout

```
src/
  retrieval/          # BM25 / dense / hybrid retrievers + HotPotQA loader + registry
    bm25.py dense.py hybrid.py reranker.py hotpotqa.py pipeline.py base.py registry.py
  generation/         # Model loading + QA prompting (registry-based)
    loader.py qa.py registry.py
  autorater/          # Sufficient-context classifier: strategies, chunking, OR-aggregation
    strategies.py aggregation.py parsing.py registry.py
  confidence/         # P(Correct) estimation (currently empty — to be filled)
  gate/               # Selective-generation gate
    features.py models.py calibration.py conformal.py selective.py plots.py
  evaluation.py       # EM/F1, abstain detection, correct/abstain/hallucinate categorization
  prompts.py          # Prompt templates
  confidence.py       # Legacy flat module (being migrated into confidence/)
  config.py           # YAML config loader (NOT git-tracked — currently untracked, see status)
  utils.py
configs/
  default.yaml
  experiments/        # per-experiment overrides
run/pipeline/         # workflow scripts
notebooks/main_pipeline.ipynb
prompts/              # externalized prompt templates
reports/              # baseline_report.md etc.
results/              # experiment outputs
data/                 # datasets
tests/                # (empty — needs pytest scaffolding)
```

---

## Architecture conventions

Follow ML-project style from global `~/.claude/rules/coding-style.md`:

- **Factory & Registry** pattern for pluggable components. `registry.py` modules already exist in `retrieval/`, `generation/`, `autorater/` — use `@register_*` decorators and `*Factory(name)` getters. Do the same for new retriever / generator / rater variants.
- **Config-driven init**: components should take a single `cfg` object; read hyperparameters from it, no hardcoding.
- **Files 200–400 lines**; split over the subpackage when growing.
- **Immutable config**: prefer `@dataclass(frozen=True)` for config objects.
- **Type hints** on every function; use `logging.getLogger(__name__)` instead of `print`.
- **Import order**: stdlib → third-party → local.

---

## Pipeline

1. **Dataset**: HotPotQA dev subset (500 examples default)
2. **Retrieval**: BM25 over distractor passages, top-5, ~4096-token context budget
3. **Generation**: Mistral-7B-Instruct, greedy, JSON output `{answer, confidence}`
4. **Evaluation**: EM / F1, abstain detection, categorization (correct / abstain / hallucinate)
5. **Autorater**: chunk the context, LLM-judge each chunk for sufficiency, OR-aggregate
6. **Gate**: logistic regression on `(confidence, sufficiency)` with threshold sweep
7. **Reporting**: selective accuracy-coverage curves, baseline vs. proposed

Target metric: **higher selective accuracy at fixed coverage** vs. confidence-only baseline; higher hallucination rate when sufficiency=0.

---

## Reproducibility

Per `~/.claude/rules/experiment-reproducibility.md`:

- Seed with `set_seed(42)` (`random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`, cudnn deterministic)
- Save resolved config + `pip freeze` alongside every experiment output
- Checkpoint naming: `best_model.pt`, `checkpoint_epoch_*.pt`, `checkpoint_latest.pt`
- Record GPU model, CUDA version, torch version at run start
- Hash dataset file (SHA256 truncated to 12 chars) and log it

---

## Git workflow

- **Conventional Commits** (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:` — scopes: `retrieval`, `generation`, `autorater`, `gate`, `config`, `eval`, `workflow`)
- Current working branch `karim`; PRs target `main`
- Rebase for branch sync, `merge --no-ff` for integration

**Uncommitted state at setup**: modified `.gitignore`, untracked `src/config.py` — investigate before committing; `config.py` may belong in-tree or be auto-generated from `configs/`.

---

## Security

- Never commit `.env`, API tokens, or HF auth tokens. Mistral-7B-Instruct was chosen specifically to avoid gated-model credentials in this repo.
- `settings.json`, `*.key`, `*.pem`, `credentials.json` are blocked by the global `security-guard.js` hook.

---

## Verification

Before committing non-trivial changes:

```bash
uv run ruff check .
uv run mypy src/
uv run pytest        # tests/ is currently empty — scaffold as needed
```

Colab smoke test: run `notebooks/main_pipeline.ipynb` top-to-bottom on a GPU runtime.

---

## Common commands in this repo

| Intent | Command |
|--------|---------|
| Install deps | `pip install -r requirements.txt` or `uv pip install -r requirements.txt` |
| Run full pipeline | open `notebooks/main_pipeline.ipynb` in Colab |
| Run experiment script | `python -m run.pipeline.<script>` |
| Inspect baseline | see `reports/baseline_report.md` |

---

## Open tasks / known gaps

- `src/confidence/` package is empty; legacy `src/confidence.py` should be migrated in and split by strategy.
- `tests/` is empty — add pytest tests for retrieval, autorater parsing/aggregation, gate calibration.
- `src/config.py` is untracked; decide whether it belongs in git or should be generated.
- Hydra is in the global stack preference but not wired in yet — current config is plain YAML via `src/config.py`.
