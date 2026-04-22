"""QA prompting, JSON parsing and batched answer generation."""

from __future__ import annotations

import json
from typing import Any

from tqdm import tqdm

from src.prompts import load_prompt
from src.utils import generate_text


def format_prompt(question: str, context: str) -> str:
    return load_prompt("qa").format(context=context, question=question)


def parse_llm_response(raw_output: str) -> dict[str, Any]:
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
    return {"answer": raw_output.strip(), "confidence": 0.1}


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
