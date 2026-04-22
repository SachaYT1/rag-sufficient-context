"""Smoke tests for Factory/Registry surfaces."""

from __future__ import annotations

import pytest

from src.autorater.registry import AUTORATER_REGISTRY, build_autorater
from src.confidence.registry import CONFIDENCE_REGISTRY, build_confidence_estimator
from src.gate.models import GATE_REGISTRY, build_gate
from src.generation.registry import MODEL_REGISTRY, resolve_hf_id
from src.retrieval.registry import RETRIEVER_REGISTRY, build_retriever


def test_model_registry_known_names() -> None:
    assert "qwen2.5-3b-instruct" in MODEL_REGISTRY
    assert "qwen2.5-7b-instruct" in MODEL_REGISTRY
    assert "mistral-7b-instruct-v0.3" in MODEL_REGISTRY
    assert resolve_hf_id("qwen2.5-3b-instruct") == "Qwen/Qwen2.5-3B-Instruct"
    assert resolve_hf_id("Qwen/Qwen2.5-3B-Instruct") == "Qwen/Qwen2.5-3B-Instruct"


def test_retriever_registry_builds_bm25() -> None:
    assert "bm25" in RETRIEVER_REGISTRY
    retriever = build_retriever("bm25")
    assert retriever.name == "bm25"


def test_autorater_registry_builds_basic() -> None:
    for key in ("basic", "cot", "fewshot"):
        assert key in AUTORATER_REGISTRY
    rater = build_autorater("basic")
    assert rater.name == "basic"


def test_confidence_registry_builds_inline() -> None:
    for key in ("inline", "self_report", "p_true", "token_entropy", "self_consistency", "semantic_entropy"):
        assert key in CONFIDENCE_REGISTRY
    est = build_confidence_estimator("inline")
    assert est.name == "inline"


def test_gate_registry_builds_logreg() -> None:
    assert "logistic_regression" in GATE_REGISTRY
    gate = build_gate("logistic_regression")
    assert hasattr(gate, "fit")


def test_unknown_key_raises() -> None:
    with pytest.raises(ValueError):
        build_retriever("unknown_retriever")
