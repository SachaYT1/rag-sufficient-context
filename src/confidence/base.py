"""Confidence estimator abstractions and JSON parsing helpers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


def parse_probability_response(raw_output: str, key: str) -> float:
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
    name: str = "base"

    @abstractmethod
    def estimate(
        self,
        example: dict[str, Any],
        model: Any,
        tokenizer: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Return (confidence, diagnostics)."""
