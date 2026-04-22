"""Tests for typed pipeline config."""

from __future__ import annotations

from src.config import dict_to_pipeline_config, load_config_typed, pipeline_config_to_dict


def test_load_default_yaml() -> None:
    cfg = load_config_typed("configs/default.yaml")
    assert cfg.experiment.name == "rag_sufficient_context"
    assert cfg.dataset.num_examples >= 1
    assert cfg.retrieval.top_k >= 1


def test_unknown_keys_ignored() -> None:
    raw = {"experiment": {"name": "x", "unexpected": 42}}
    cfg = dict_to_pipeline_config(raw)
    assert cfg.experiment.name == "x"


def test_roundtrip() -> None:
    cfg = load_config_typed("configs/default.yaml")
    as_dict = pipeline_config_to_dict(cfg)
    assert as_dict["retrieval"]["top_k"] == cfg.retrieval.top_k
