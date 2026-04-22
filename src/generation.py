"""LLM generation with config-driven model loading and token-level statistics."""

from __future__ import annotations

import json
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import generate_text


QA_PROMPT_TEMPLATE = """You are a careful question-answering assistant.

Answer the question using only the provided context.
If the context is not sufficient to answer the question, answer exactly: "I don't know".

Context:
{context}

Question:
{question}

Return ONLY valid JSON in this format:
{{"answer": "your answer here", "confidence": 0.0}}

Rules:
- confidence must be between 0.0 and 1.0
- if the context is insufficient, set answer to "I don't know" and confidence to 0.0
- do not include any extra commentary outside the JSON
"""


def _resolve_model_dtype(dtype_name: str | None):
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
):
    """Load model and tokenizer using config as the single source of truth."""
    model_config = model_config or {}
    resolved_model_name = model_name or model_config.get("model_name") or model_config.get("model")
    if not resolved_model_name:
        raise ValueError("Model name must be provided via `model_name` or `model_config`.")

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_dtype = _resolve_model_dtype(model_config.get("torch_dtype", "float16"))
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_name,
        torch_dtype=resolved_dtype,
        device_map=device or model_config.get("device_map", "auto"),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    model.eval()
    return model, tokenizer


def format_prompt(question: str, context: str) -> str:
    """Format the QA prompt."""
    return QA_PROMPT_TEMPLATE.format(context=context, question=question)


def parse_llm_response(raw_output: str) -> dict[str, Any]:
    """Parse JSON response from LLM output."""
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw_output[start:end])
            confidence = float(parsed.get("confidence", 0.0))
            return {
                "answer": str(parsed.get("answer", "")).strip(),
                "confidence": max(0.0, min(1.0, confidence)),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return {
        "answer": raw_output.strip(),
        "confidence": 0.1,
    }


def generate_answer(
    question: str,
    context: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    """Generate a single answer and preserve generation diagnostics."""
    prompt = format_prompt(question, context)
    generation_details = generate_text(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        greedy=(temperature == 0.0),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        return_details=True,
    )
    raw_output = generation_details["text"]
    parsed = parse_llm_response(raw_output)
    parsed["raw_output"] = raw_output
    parsed["prompt"] = prompt
    parsed["generation_stats"] = generation_details
    parsed["token_entropy_proxy"] = generation_details.get("token_entropy_proxy")
    parsed["mean_token_probability"] = generation_details.get("mean_token_probability")
    return parsed


def generate_answers_batch(
    examples: list[dict[str, Any]],
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
) -> list[dict[str, Any]]:
    """Generate answers for all examples."""
    results: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc="Generating answers"):
        answer_data = generate_answer(
            question=ex["question"],
            context=ex["context"],
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        results.append(
            {
                **ex,
                "prediction": answer_data["answer"],
                "confidence": answer_data["confidence"],
                "raw_output": answer_data["raw_output"],
                "prompt": answer_data["prompt"],
                "generation_stats": answer_data["generation_stats"],
                "token_entropy_proxy": answer_data["token_entropy_proxy"],
                "mean_token_probability": answer_data["mean_token_probability"],
            }
        )
    return results