"""Registry of supported generation models with loader factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

ModelLoader = Callable[..., tuple[Any, Any]]

MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class ModelEntry:
    hf_id: str
    family: str
    recommended_dtype: str
    notes: str = ""


def register_model(
    name: str,
    hf_id: str,
    family: str,
    recommended_dtype: str = "float16",
    notes: str = "",
) -> None:
    MODEL_REGISTRY[name] = {
        "entry": ModelEntry(hf_id=hf_id, family=family, recommended_dtype=recommended_dtype, notes=notes),
    }


register_model(
    "mistral-7b-instruct-v0.3",
    hf_id="mistralai/Mistral-7B-Instruct-v0.3",
    family="mistral",
    recommended_dtype="float16",
    notes="Baseline used in the original project.",
)
register_model(
    "llama-3.1-8b-instruct",
    hf_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    family="llama",
    recommended_dtype="bfloat16",
    notes="Matches ICLR'25 sufficient-context paper (gated HF access).",
)
register_model(
    "qwen2.5-3b-instruct",
    hf_id="Qwen/Qwen2.5-3B-Instruct",
    family="qwen",
    recommended_dtype="bfloat16",
    notes="Small generator for asymmetric setup; fast on Colab T4.",
)
register_model(
    "qwen2.5-7b-instruct",
    hf_id="Qwen/Qwen2.5-7B-Instruct",
    family="qwen",
    recommended_dtype="bfloat16",
    notes="Strong open-weights generator and autorater.",
)
register_model(
    "phi-3.5-mini-instruct",
    hf_id="microsoft/Phi-3.5-mini-instruct",
    family="phi",
    recommended_dtype="bfloat16",
    notes="Compact reasoning-strong baseline.",
)


def resolve_hf_id(name_or_hf_id: str) -> str:
    """Accept both short registry names and full HF ids."""
    if name_or_hf_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[name_or_hf_id]["entry"].hf_id
    return name_or_hf_id
