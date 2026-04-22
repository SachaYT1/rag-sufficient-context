"""Typed, immutable configuration objects for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "rag_sufficient_context"
    output_dir: str = "results"
    seed: int = 42


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "hotpotqa"
    split: str = "validation"
    num_examples: int = 500
    seed: int = 42


@dataclass(frozen=True)
class RetrievalConfig:
    method: str = "bm25_distractor_rerank"
    top_k: int = 5
    max_context_tokens: int = 4096
    dense_model: str = "intfloat/e5-base-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    hybrid_weights: tuple[float, float] = (0.5, 0.5)


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0


@dataclass(frozen=True)
class AutoraterConfig:
    method: str = "basic"
    chunk_size_tokens: int = 1400
    aggregation: str = "support_all_required"
    max_new_tokens: int = 96
    model_name: str | None = None  # None means reuse generator model


@dataclass(frozen=True)
class ConfidenceConfig:
    method: str = "self_report"
    max_new_tokens: int = 64
    num_samples: int = 5
    sample_temperature: float = 0.7
    sample_top_p: float = 0.95


@dataclass(frozen=True)
class GateConfig:
    model: str = "logistic_regression"
    threshold_steps: int = 50
    conformal_alpha: float = 0.1


@dataclass(frozen=True)
class EvaluationConfig:
    f1_threshold: float = 0.5


@dataclass(frozen=True)
class PipelineConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    autorater: AutoraterConfig = field(default_factory=AutoraterConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _filter_known_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys to keep dataclass construction forward-compatible."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


def dict_to_pipeline_config(data: dict[str, Any]) -> PipelineConfig:
    """Build a PipelineConfig from a raw YAML/dict payload."""
    section_map: dict[str, type] = {
        "experiment": ExperimentConfig,
        "dataset": DatasetConfig,
        "retrieval": RetrievalConfig,
        "generation": GenerationConfig,
        "autorater": AutoraterConfig,
        "confidence": ConfidenceConfig,
        "gate": GateConfig,
        "evaluation": EvaluationConfig,
    }
    kwargs: dict[str, Any] = {}
    for key, section_cls in section_map.items():
        section_data = data.get(key, {}) or {}
        kwargs[key] = section_cls(**_filter_known_fields(section_cls, section_data))
    return PipelineConfig(**kwargs)


def load_config_typed(
    config_path: str | Path = "configs/default.yaml",
) -> PipelineConfig:
    """Load YAML and construct a typed PipelineConfig."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return dict_to_pipeline_config(raw)


def pipeline_config_to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    """Convert typed config back to a plain dictionary for logging/metadata."""

    def _to_dict(value: Any) -> Any:
        if is_dataclass(value):
            return {f.name: _to_dict(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, (list, tuple)):
            return [_to_dict(v) for v in value]
        if isinstance(value, dict):
            return {k: _to_dict(v) for k, v in value.items()}
        return value

    return _to_dict(cfg)
