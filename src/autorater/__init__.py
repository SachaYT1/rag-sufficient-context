"""Autorater subsystem."""

from __future__ import annotations

from src.autorater.aggregation import aggregate_passage_ratings
from src.autorater.parsing import parse_autorater_response
from src.autorater.pipeline import rate_all_examples, rate_single_context, rate_sufficiency
from src.autorater.registry import AUTORATER_REGISTRY, build_autorater, register_autorater
from src.autorater.strategies import BaseAutorater, PromptAutorater, SelfConsistencyAutorater

__all__ = [
    "AUTORATER_REGISTRY",
    "BaseAutorater",
    "PromptAutorater",
    "SelfConsistencyAutorater",
    "aggregate_passage_ratings",
    "build_autorater",
    "parse_autorater_response",
    "rate_all_examples",
    "rate_single_context",
    "rate_sufficiency",
    "register_autorater",
]
