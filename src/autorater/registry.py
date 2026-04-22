"""Factory for autorater strategies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.autorater.strategies import BaseAutorater, PromptAutorater, SelfConsistencyAutorater

AUTORATER_REGISTRY: dict[str, Callable[..., BaseAutorater]] = {}


def register_autorater(name: str) -> Callable[[Callable[..., BaseAutorater]], Callable[..., BaseAutorater]]:
    def decorator(fn: Callable[..., BaseAutorater]) -> Callable[..., BaseAutorater]:
        AUTORATER_REGISTRY[name] = fn
        return fn

    return decorator


@register_autorater("basic")
def _basic(**kwargs: Any) -> BaseAutorater:
    return PromptAutorater(prompt_name="autorater_basic", name="basic")


@register_autorater("cot")
def _cot(**kwargs: Any) -> BaseAutorater:
    return PromptAutorater(prompt_name="autorater_cot", name="cot")


@register_autorater("fewshot")
def _fewshot(**kwargs: Any) -> BaseAutorater:
    return PromptAutorater(prompt_name="autorater_fewshot", name="fewshot")


@register_autorater("self_consistency")
def _sc(**kwargs: Any) -> BaseAutorater:
    return SelfConsistencyAutorater(
        prompt_name=kwargs.get("prompt_name", "autorater_basic"),
        num_samples=kwargs.get("num_samples", 5),
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.95),
    )


def build_autorater(name: str = "basic", **kwargs: Any) -> BaseAutorater:
    if name not in AUTORATER_REGISTRY:
        raise ValueError(
            f"Unknown autorater '{name}'. Available: {sorted(AUTORATER_REGISTRY)}"
        )
    return AUTORATER_REGISTRY[name](**kwargs)
