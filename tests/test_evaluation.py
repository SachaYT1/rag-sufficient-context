"""Evaluation metrics — QA and abstention detection."""

from __future__ import annotations

from src.evaluation import (
    categorize_response,
    exact_match,
    f1_score,
    is_abstention,
    normalize_answer,
)


def test_normalize_answer() -> None:
    assert normalize_answer("The cat.") == "cat"
    assert normalize_answer("A  Dog!") == "dog"


def test_exact_match() -> None:
    assert exact_match("Paris", "the Paris.") == 1.0
    assert exact_match("London", "Paris") == 0.0


def test_f1_score() -> None:
    # pred="paris france" (2 tokens), gold="paris" (1 token) -> P=0.5, R=1.0, F1=2/3
    assert abs(f1_score("Paris, France", "Paris") - 2 / 3) < 1e-9
    assert f1_score("foo", "bar") == 0.0


def test_abstention_detection() -> None:
    assert is_abstention("I don't know")
    assert is_abstention("insufficient information in context")
    assert not is_abstention("Paris is the capital")


def test_categorize_response() -> None:
    assert categorize_response("I don't know", "Paris") == "abstain"
    assert categorize_response("Paris", "Paris") == "correct"
    assert categorize_response("Madrid", "Paris") == "hallucinate"
