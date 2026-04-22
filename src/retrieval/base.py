"""Retriever abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """Abstract retriever that scores passages for a given query."""

    name: str = "base"

    @abstractmethod
    def score(
        self,
        query: str,
        passages: list[dict[str, Any]],
    ) -> list[float]:
        """Return a list of relevance scores aligned with ``passages``."""
