"""Cross-encoder reranker wrapper applied to a candidate top-N list."""

from __future__ import annotations

from typing import Any

from src.retrieval.base import BaseRetriever


class CrossEncoderReranker(BaseRetriever):
    """Cross-encoder reranker (e.g. BAAI/bge-reranker-base)."""

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker."
                ) from exc
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, passages: list[dict[str, Any]]) -> list[float]:
        model = self._load()
        pairs = [(query, p["text"]) for p in passages]
        scores = model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
