"""Unit tests for QA JSON response parsing."""

from __future__ import annotations

from src.generation.qa import parse_llm_response


def test_parse_valid_json() -> None:
    out = parse_llm_response('{"answer": "Paris", "confidence": 0.8}')
    assert out == {"answer": "Paris", "confidence": 0.8}


def test_parse_embedded_json() -> None:
    out = parse_llm_response('noise {"answer": "42", "confidence": 1.2} trailing')
    assert out["answer"] == "42"
    assert out["confidence"] == 1.0  # clamped to [0, 1]


def test_parse_malformed_returns_fallback() -> None:
    out = parse_llm_response("hello world")
    assert out["answer"] == "hello world"
    assert 0.0 <= out["confidence"] <= 1.0


def test_parse_negative_confidence_clamped() -> None:
    out = parse_llm_response('{"answer": "x", "confidence": -0.5}')
    assert out["confidence"] == 0.0
