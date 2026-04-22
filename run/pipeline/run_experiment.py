"""Single-config experiment runner.

Usage:
    python -m run.pipeline.run_experiment configs/experiments/qwen3b_bm25.yaml

This runner is environment-aware: heavy imports (transformers, torch) only
happen when the corresponding stage actually runs. When model loading fails
(e.g., no GPU, no HF access), the runner logs a clear message and exits
non-zero so a matrix orchestrator can skip that cell.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.autorater import build_autorater, rate_all_examples
from src.config import load_config_typed, pipeline_config_to_dict
from src.confidence import estimate_confidence_batch
from src.data import load_dataset_by_name
from src.evaluation import evaluate_all
from src.gate import compute_selective_curves, conformal_selective
from src.generation import generate_answers_batch, load_model
from src.retrieval import build_retriever, build_retrieval_pipeline
from src.utils import cache_results, save_run_metadata, set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _output_dir(cfg) -> Path:
    out = Path(cfg.experiment.output_dir) / cfg.experiment.name
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(config_path: str) -> dict[str, Any]:
    cfg = load_config_typed(config_path)
    set_global_seed(cfg.experiment.seed)
    output_dir = _output_dir(cfg)

    logger.info("Loading dataset %s (%d examples)", cfg.dataset.name, cfg.dataset.num_examples)
    examples = load_dataset_by_name(
        cfg.dataset.name,
        split=cfg.dataset.split,
        num_examples=cfg.dataset.num_examples,
        seed=cfg.dataset.seed,
    )

    logger.info("Loading generator %s", cfg.generation.model_name)
    gen_model, gen_tokenizer = load_model(
        model_name=cfg.generation.model_name,
        model_config=pipeline_config_to_dict(cfg)["generation"],
    )

    logger.info("Retrieval method=%s top_k=%d", cfg.retrieval.method, cfg.retrieval.top_k)
    retriever = build_retriever(
        cfg.retrieval.method,
        dense_model=cfg.retrieval.dense_model,
        reranker_model=cfg.retrieval.reranker_model,
        hybrid_weights=cfg.retrieval.hybrid_weights,
    )
    retrieved = build_retrieval_pipeline(
        examples=examples,
        top_k=cfg.retrieval.top_k,
        max_context_tokens=cfg.retrieval.max_context_tokens,
        tokenizer=gen_tokenizer,
        retriever=retriever,
    )
    cache_results(retrieved, str(output_dir / "retrieval.json"))

    logger.info("Generating answers")
    generations = generate_answers_batch(
        examples=retrieved,
        model=gen_model,
        tokenizer=gen_tokenizer,
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        top_k=cfg.generation.top_k,
        repetition_penalty=cfg.generation.repetition_penalty,
    )

    logger.info("Estimating confidence (%s)", cfg.confidence.method)
    with_conf = estimate_confidence_batch(
        examples=generations,
        model=gen_model,
        tokenizer=gen_tokenizer,
        method=cfg.confidence.method,
        method_config=pipeline_config_to_dict(cfg)["confidence"],
    )

    # Optional asymmetric autorater model
    autorater_model_name = cfg.autorater.model_name or cfg.generation.model_name
    if autorater_model_name != cfg.generation.model_name:
        logger.info("Loading asymmetric autorater model %s", autorater_model_name)
        ar_model, ar_tokenizer = load_model(
            model_name=autorater_model_name,
            model_config=pipeline_config_to_dict(cfg)["generation"],
        )
    else:
        ar_model, ar_tokenizer = gen_model, gen_tokenizer

    autorater = build_autorater(cfg.autorater.method)
    logger.info("Rating sufficiency (%s)", cfg.autorater.method)
    rated = rate_all_examples(
        examples=with_conf,
        model=ar_model,
        tokenizer=ar_tokenizer,
        chunk_size_tokens=cfg.autorater.chunk_size_tokens,
        aggregation=cfg.autorater.aggregation,
        max_new_tokens=cfg.autorater.max_new_tokens,
        autorater=autorater,
    )

    logger.info("Evaluating")
    evaluation = evaluate_all(
        examples=rated,
        f1_threshold=cfg.evaluation.f1_threshold,
        output_dir=str(output_dir),
        config=pipeline_config_to_dict(cfg),
        model_name=cfg.generation.model_name,
    )

    logger.info("Computing selective curves (gate=%s)", cfg.gate.model)
    per_example = evaluation["per_example"]
    curves = compute_selective_curves(per_example, gate_name=cfg.gate.model)
    cache_results(curves, str(output_dir / "selective_curves.json"))

    conformal = None
    if curves.get("gate_scores") and curves.get("labels"):
        import numpy as np

        conformal = conformal_selective(
            scores=np.asarray(curves["gate_scores"], dtype=float),
            labels=np.asarray(curves["labels"], dtype=float),
            alpha=cfg.gate.conformal_alpha,
            calib_fraction=0.3,
            seed=cfg.experiment.seed,
        )
        cache_results(conformal, str(output_dir / "conformal.json"))

    save_run_metadata(
        output_dir=str(output_dir),
        config=pipeline_config_to_dict(cfg),
        model_name=cfg.generation.model_name,
        extra={"autorater_model": autorater_model_name},
    )

    summary = {
        "experiment": cfg.experiment.name,
        "metrics": evaluation["metrics"],
        "baseline_aurc": curves.get("baseline", {}).get("aurc"),
        "proposed_aurc": curves.get("proposed", {}).get("aurc"),
        "gate_meta": curves.get("gate_meta"),
        "conformal": conformal,
    }
    cache_results(summary, str(output_dir / "summary.json"))
    logger.info("Done. Summary: %s", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single RAG experiment.")
    parser.add_argument("config_path", help="Path to experiment YAML.")
    args = parser.parse_args()
    try:
        run(args.config_path)
        return 0
    except Exception as exc:  # pragma: no cover - runtime failure path
        logger.exception("Experiment failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
