"""HuggingFace model loading with config-driven dtype resolution."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation.registry import resolve_hf_id


def resolve_model_dtype(dtype_name: str | None) -> Any:
    if not dtype_name or dtype_name == "auto":
        return "auto"
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype_map[dtype_name]


def load_model(
    model_name: str | None = None,
    device: str | None = None,
    model_config: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Load a model+tokenizer pair from the registry or a raw HF id."""
    model_config = model_config or {}
    raw_name = model_name or model_config.get("model_name") or model_config.get("model")
    if not raw_name:
        raise ValueError("Model name must be provided via `model_name` or `model_config`.")

    hf_id = resolve_hf_id(raw_name)
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_model_dtype(model_config.get("torch_dtype", "float16"))
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map=device or model_config.get("device_map", "auto"),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    model.eval()
    return model, tokenizer
