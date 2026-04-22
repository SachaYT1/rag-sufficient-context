"""Generation subsystem.

Heavy dependencies (``transformers``) are imported lazily via ``__getattr__``
so lightweight tests that only need JSON parsing or prompt templates do not
pay the import cost.
"""

from __future__ import annotations

from typing import Any

from src.generation.qa import (
    format_prompt,
    generate_answer,
    generate_answers_batch,
    parse_llm_response,
)
from src.generation.registry import MODEL_REGISTRY, register_model, resolve_hf_id
from src.prompts import load_prompt as _load_prompt

QA_PROMPT_TEMPLATE = _load_prompt("qa")


def __getattr__(name: str) -> Any:
    if name in {"load_model", "resolve_model_dtype"}:
        from src.generation.loader import load_model, resolve_model_dtype

        return {"load_model": load_model, "resolve_model_dtype": resolve_model_dtype}[name]
    raise AttributeError(f"module 'src.generation' has no attribute {name!r}")


__all__ = [
    "MODEL_REGISTRY",
    "QA_PROMPT_TEMPLATE",
    "format_prompt",
    "generate_answer",
    "generate_answers_batch",
    "load_model",
    "parse_llm_response",
    "register_model",
    "resolve_hf_id",
    "resolve_model_dtype",
]
