"""Confidence estimators with a common interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from tqdm import tqdm

from src.generation import generate_answer
from src.utils import generate_text


CONFIDENCE_PROMPT_TEMPLATE = """You previously answered a question using the context below.

Context:
{context}

Question:
{question}

Answer:
{answer}

Estimate the probability that the answer is correct based only on the context.
Return ONLY valid JSON:
{{"confidence": 0.0}}
"""


P_TRUE_PROMPT_TEMPLATE = """Question:
{question}

Context:
{context}

Candidate answer:
{answer}

Is the candidate answer supported by the context?
Return ONLY valid JSON:
{{"p_true": 0.0}}

Where p_true is between 0.0 and 1.0.
"""


def _parse_probability_response(raw_output: str, key: str) -> float:
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw_output[start:end])
            value = float(parsed.get(key, 0.0))
            return max(0.0, min(1.0, value))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return 0.0


class BaseConfidenceEstimator(ABC):
    """Abstract confidence estimator."""

    @abstractmethod
    def estimate(
        self,
        example: dict[str, Any],
        model,
        tokenizer,
    ) -> tuple[float, dict[str, Any]]:
        """Return confidence score and optional diagnostic metadata."""


class InlineConfidenceEstimator(BaseConfidenceEstimator):
    """Use the confidence already produced by generation."""

    def estimate(
        self,
        example: dict[str, Any],
        model,
        tokenizer,
    ) -> tuple[float, dict[str, Any]]:
        return float(example.get("confidence", 0.0)), {
            "confidence_method": "inline",
        }


class SelfReportConfidenceEstimator(BaseConfidenceEstimator):
    """Ask the model to self-report the probability of correctness."""

    def __init__(self, max_new_tokens: int = 64):
        self.max_new_tokens = max_new_tokens

    def estimate(
        self,
        example: dict[str, Any],
        model,
        tokenizer,
    ) -> tuple[float, dict[str, Any]]:
        prompt = CONFIDENCE_PROMPT_TEMPLATE.format(
            context=example["context"],
            question=example["question"],
            answer=example["prediction"],
        )
        raw_output = generate_text(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            greedy=True,
        )
        confidence = _parse_probability_response(raw_output, "confidence")
        return confidence, {
            "confidence_method": "self_report",
            "confidence_raw_output": raw_output,
        }


class TokenEntropyConfidenceEstimator(BaseConfidenceEstimator):
    """Convert generation entropy proxy to confidence."""

    def estimate(
        self,
        example: dict[str, Any],
        model,
        tokenizer,
    ) -> tuple[float, dict[str, Any]]:
        entropy_proxy = example.get("token_entropy_proxy")
        if entropy_proxy is None:
            confidence = float(example.get("mean_token_probability", 0.0))
        else:
            confidence = float(np.exp(-float(entropy_proxy)))
        confidence = max(0.0, min(1.0, confidence))
        return confidence, {
            "confidence_method": "token_entropy",
            "entropy_proxy": entropy_proxy,
        }


class PTrueConfidenceEstimator(BaseConfidenceEstimator):
    """Directly estimate whether the candidate answer is supported by the context."""

    def __init__(self, max_new_tokens: int = 64):
        self.max_new_tokens = max_new_tokens

    def estimate(
        self,
        example: dict[str, Any],
        model,
        tokenizer,
    ) -> tuple[float, dict[str, Any]]:
        prompt = P_TRUE_PROMPT_TEMPLATE.format(
            question=example["question"],
            context=example["context"],
            answer=example["prediction"],
        )
        raw_output = generate_text(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            greedy=True,
        )
        confidence = _parse_probability_response(raw_output, "p_true")
        return confidence, {
            "confidence_method": "p_true",
            "confidence_raw_output": raw_output,
        }


class SelfConsistencyConfidenceEstimator(BaseConfidenceEstimator):
    """Estimate confidence from agreement across multiple stochastic samples."""

    def __init__(
        self,
        num_samples: int = 5,
        sample_temperature: float = 0.7,
        sample_top_p: float = 0.95,
        max_new_tokens: int = 256,
    ):
        self.num_samples = num_samples
        self.sample_temperature = sample_temperature
        self.sample_top_p = sample_top_p
        self.max_new_tokens = max_new_tokens

    def estimate(
        self,
        example: dict[str, Any],
        model,
        tokenizer,
    ) -> tuple[float, dict[str, Any]]:
        answers: list[str] = []
        for _ in range(self.num_samples):
            sample = generate_answer(
                question=example["question"],
                context=example["context"],
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.sample_temperature,
                top_p=self.sample_top_p,
                top_k=50,
            )
            answers.append(sample["answer"].strip().lower())

        if not answers:
            return 0.0, {"confidence_method": "self_consistency", "samples": []}

        majority_answer = max(set(answers), key=answers.count)
        confidence = answers.count(majority_answer) / len(answers)
        return confidence, {
            "confidence_method": "self_consistency",
            "samples": answers,
            "majority_answer": majority_answer,
        }


def build_confidence_estimator(
    method: str = "inline",
    config: dict[str, Any] | None = None,
) -> BaseConfidenceEstimator:
    """Factory for confidence estimators."""
    config = config or {}
    if method == "inline":
        return InlineConfidenceEstimator()
    if method == "self_report":
        return SelfReportConfidenceEstimator(
            max_new_tokens=config.get("max_new_tokens", 64),
        )
    if method == "token_entropy":
        return TokenEntropyConfidenceEstimator()
    if method == "p_true":
        return PTrueConfidenceEstimator(
            max_new_tokens=config.get("max_new_tokens", 64),
        )
    if method == "self_consistency":
        return SelfConsistencyConfidenceEstimator(
            num_samples=config.get("num_samples", 5),
            sample_temperature=config.get("sample_temperature", 0.7),
            sample_top_p=config.get("sample_top_p", 0.95),
            max_new_tokens=config.get("max_new_tokens", 256),
        )
    raise ValueError(f"Unsupported confidence method: {method}")


def estimate_confidence_batch(
    examples: list[dict[str, Any]],
    model,
    tokenizer,
    use_inline: bool = True,
    method: str | None = None,
    method_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Estimate confidence for a batch with interchangeable estimators."""
    resolved_method = method or ("inline" if use_inline else "self_report")
    estimator = build_confidence_estimator(resolved_method, method_config)

    results: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc=f"Estimating confidence ({resolved_method})"):
        confidence, diagnostics = estimator.estimate(ex, model, tokenizer)
        results.append(
            {
                **ex,
                "confidence": float(confidence),
                "confidence_method": diagnostics.get("confidence_method", resolved_method),
                "confidence_diagnostics": diagnostics,
            }
        )
    return results