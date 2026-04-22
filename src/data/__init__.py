"""Dataset loaders with a uniform output schema."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.retrieval.hotpotqa import load_hotpotqa

DATASET_REGISTRY: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "hotpotqa": load_hotpotqa,
}


def register_dataset(name: str) -> Callable:
    def decorator(fn: Callable[..., list[dict[str, Any]]]):
        DATASET_REGISTRY[name] = fn
        return fn

    return decorator


def load_dataset_by_name(name: str, **kwargs: Any) -> list[dict[str, Any]]:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](**kwargs)


# NQ-Open is registered lazily when its module is imported.
from src.data.nq_open import load_nq_open  # noqa: E402,F401
