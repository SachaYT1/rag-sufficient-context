"""Shared utilities: data loading, caching, tokenization helpers."""

import json
import os
import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def cache_results(data: Any, cache_path: str) -> None:
    """Save results to JSON cache file."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cached(cache_path: str) -> Any | None:
    """Load results from JSON cache file, or None if not found."""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return None


def make_cache_key(*args: str) -> str:
    """Create a deterministic cache key from arguments."""
    combined = "|".join(str(a) for a in args)
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def truncate_text_to_tokens(text: str, max_tokens: int, tokenizer) -> str:
    """Truncate text to a maximum number of tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def chunk_text(text: str, chunk_size_tokens: int, tokenizer) -> list[str]:
    """Split text into chunks of approximately chunk_size_tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size_tokens):
        chunk_tokens = tokens[i : i + chunk_size_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks
