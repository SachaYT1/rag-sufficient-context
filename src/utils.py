"""Shared utilities for configuration, caching, tokenization and generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_results(data: Any, cache_path: str) -> None:
    """Save results to a JSON cache file."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cached(cache_path: str) -> Any | None:
    """Load cached JSON data if it exists."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def make_cache_key(*args: Any) -> str:
    """Create a deterministic cache key from a sequence of values."""
    combined = "|".join(str(a) for a in args)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()[:12]


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_package_version(package_name: str) -> str | None:
    """Return installed package version if available."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_git_commit(default: str | None = None) -> str | None:
    """Return the current git commit hash if available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return default


def build_run_metadata(
    config: dict,
    model_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a reproducibility-focused metadata payload for the current run."""
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": get_git_commit(),
        "seed": (
            config.get("experiment", {}).get("seed")
            or config.get("dataset", {}).get("seed")
        ),
        "model_name": model_name
        or config.get("generation", {}).get("model_name")
        or config.get("generation", {}).get("model"),
        "config": config,
        "package_versions": {
            "torch": safe_package_version("torch"),
            "transformers": safe_package_version("transformers"),
            "datasets": safe_package_version("datasets"),
            "rank-bm25": safe_package_version("rank-bm25"),
            "scikit-learn": safe_package_version("scikit-learn"),
            "numpy": safe_package_version("numpy"),
            "pandas": safe_package_version("pandas"),
            "matplotlib": safe_package_version("matplotlib"),
            "pyyaml": safe_package_version("PyYAML"),
            "tqdm": safe_package_version("tqdm"),
            "accelerate": safe_package_version("accelerate"),
        },
    }
    if extra:
        metadata["extra"] = extra
    return metadata


def save_run_metadata(
    output_dir: str,
    config: dict,
    model_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write run_metadata.json into the output directory and return it."""
    metadata = build_run_metadata(config=config, model_name=model_name, extra=extra)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def count_tokens(text: str, tokenizer) -> int:
    """Count text tokens using the provided tokenizer."""
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    tokenizer,
    return_metadata: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    """Truncate text to at most ``max_tokens`` tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    original_tokens = len(tokens)
    was_truncated = original_tokens > max_tokens
    if not was_truncated:
        if return_metadata:
            return text, {
                "context_tokens_before_truncation": original_tokens,
                "context_tokens_after_truncation": original_tokens,
                "was_truncated": False,
            }
        return text

    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    metadata = {
        "context_tokens_before_truncation": original_tokens,
        "context_tokens_after_truncation": len(truncated_tokens),
        "was_truncated": True,
    }
    if return_metadata:
        return truncated_text, metadata
    return truncated_text


def chunk_text(text: str, chunk_size_tokens: int, tokenizer) -> list[str]:
    """Fallback token-based chunking helper."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[str] = []
    for start in range(0, len(tokens), chunk_size_tokens):
        chunk_tokens = tokens[start : start + chunk_size_tokens]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
    return chunks


def split_passages_by_token_budget(
    passages: list[dict[str, Any]],
    tokenizer,
    max_tokens_per_group: int,
) -> list[list[dict[str, Any]]]:
    """Group passage dictionaries into token-bounded mini-batches."""
    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []
    current_tokens = 0

    for passage in passages:
        passage_text = passage.get("text", "")
        passage_tokens = count_tokens(passage_text, tokenizer)

        if current_group and current_tokens + passage_tokens > max_tokens_per_group:
            groups.append(current_group)
            current_group = []
            current_tokens = 0

        if passage_tokens > max_tokens_per_group:
            truncated_text, meta = truncate_text_to_tokens(
                passage_text,
                max_tokens=max_tokens_per_group,
                tokenizer=tokenizer,
                return_metadata=True,
            )
            trimmed_passage = {
                **passage,
                "text": truncated_text,
                "tokens_before_truncation": meta["context_tokens_before_truncation"],
                "tokens_after_truncation": meta["context_tokens_after_truncation"],
                "passage_was_truncated": meta["was_truncated"],
            }
            groups.append([trimmed_passage])
            continue

        current_group.append(passage)
        current_tokens += passage_tokens

    if current_group:
        groups.append(current_group)
    return groups


def _prepare_model_inputs(prompt: str, model, tokenizer):
    """Apply chat template and move tensors to the model device."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(input_text, return_tensors="pt").to(model.device)


def _extract_generated_token_logprobs(
    generated_sequences: torch.Tensor,
    scores: list[torch.Tensor],
    prompt_length: int,
) -> list[float]:
    """Extract log-probabilities for the generated sequence tokens."""
    logprobs: list[float] = []
    if not scores:
        return logprobs

    full_sequence = generated_sequences[0]
    generated_tokens = full_sequence[prompt_length:]
    for step_idx, step_scores in enumerate(scores):
        if step_idx >= len(generated_tokens):
            break
        token_id = generated_tokens[step_idx]
        step_log_probs = torch.log_softmax(step_scores[0], dim=-1)
        logprobs.append(float(step_log_probs[token_id].item()))
    return logprobs


def generate_text(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
    greedy: bool = True,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    return_details: bool = False,
) -> str | dict[str, Any]:
    """Generate text with consistent sampling semantics.

    When ``return_details=True`` the function also returns token-level statistics
    that can be reused by confidence estimators.
    """
    inputs = _prepare_model_inputs(prompt, model, tokenizer)
    prompt_length = inputs["input_ids"].shape[1]

    do_sample = (not greedy) and temperature > 0.0
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        generation_kwargs["top_k"] = top_k
    if repetition_penalty and repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = repetition_penalty

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    new_tokens = outputs.sequences[0][prompt_length:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if not return_details:
        return decoded

    token_logprobs = _extract_generated_token_logprobs(
        generated_sequences=outputs.sequences,
        scores=outputs.scores,
        prompt_length=prompt_length,
    )
    mean_logprob = float(np.mean(token_logprobs)) if token_logprobs else None
    mean_token_probability = (
        float(np.mean(np.exp(token_logprobs))) if token_logprobs else None
    )
    entropy_proxy = float(-mean_logprob) if mean_logprob is not None else None

    return {
        "text": decoded,
        "prompt_tokens": int(prompt_length),
        "generated_tokens": int(new_tokens.shape[0]),
        "token_logprobs": token_logprobs,
        "mean_logprob": mean_logprob,
        "mean_token_probability": mean_token_probability,
        "token_entropy_proxy": entropy_proxy,
        "sampling": {
            "greedy": greedy,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
        },
    }