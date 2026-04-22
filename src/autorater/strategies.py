"""Autorater strategies: basic, CoT, few-shot, self-consistency."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.autorater.parsing import parse_autorater_response
from src.prompts import load_prompt
from src.utils import generate_text


class BaseAutorater:
    name: str = "base"

    def rate(
        self,
        question: str,
        context: str,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int = 96,
    ) -> dict[str, Any]:
        raise NotImplementedError


class PromptAutorater(BaseAutorater):
    """Single-prompt autorater (basic, CoT, few-shot)."""

    def __init__(self, prompt_name: str, name: str):
        self.prompt_name = prompt_name
        self.name = name
        self._template = load_prompt(prompt_name)

    def rate(
        self,
        question: str,
        context: str,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int = 96,
    ) -> dict[str, Any]:
        prompt = self._template.format(question=question, context=context)
        raw_output = generate_text(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        return parse_autorater_response(raw_output)


class SelfConsistencyAutorater(BaseAutorater):
    """Majority-vote across stochastic samples of the basic prompt."""

    name = "self_consistency"

    def __init__(
        self,
        prompt_name: str = "autorater_basic",
        num_samples: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self._template = load_prompt(prompt_name)
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p

    def rate(
        self,
        question: str,
        context: str,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int = 96,
    ) -> dict[str, Any]:
        prompt = self._template.format(question=question, context=context)
        votes: list[bool] = []
        raw_outputs: list[str] = []
        for _ in range(self.num_samples):
            raw = generate_text(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                greedy=False,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            parsed = parse_autorater_response(raw)
            votes.append(bool(parsed["sufficient"]))
            raw_outputs.append(raw)

        majority = Counter(votes).most_common(1)[0][0]
        return {
            "sufficient": bool(majority),
            "reason": f"self_consistency votes: {votes}",
            "raw_output": raw_outputs,
            "parsed": True,
            "votes": votes,
        }
