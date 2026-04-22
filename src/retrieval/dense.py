"""Dense retriever with lazy sentence-transformers import."""

from __future__ import annotations

from typing import Any

from src.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense passage retriever using an embedding model (default: E5-base-v2).

    The model is loaded lazily so the default test environment does not require
    sentence-transformers to be installed.
    """

    name = "dense"

    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "sentence-transformers is required for DenseRetriever."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @staticmethod
    def _prefix_query(q: str) -> str:
        return f"query: {q}"

    @staticmethod
    def _prefix_passage(p: str) -> str:
        return f"passage: {p}"

    def score(self, query: str, passages: list[dict[str, Any]]) -> list[float]:
        import numpy as np

        model = self._load()
        q_emb = model.encode([self._prefix_query(query)], normalize_embeddings=True)
        p_texts = [self._prefix_passage(p["text"]) for p in passages]
        p_emb = model.encode(p_texts, normalize_embeddings=True, batch_size=32)
        sims = np.asarray(p_emb) @ np.asarray(q_emb).T
        return [float(s) for s in sims[:, 0]]
