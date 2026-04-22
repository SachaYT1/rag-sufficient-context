"""Individual confidence estimators sharing a common interface."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.confidence.base import BaseConfidenceEstimator, parse_probability_response
from src.prompts import load_prompt
from src.utils import generate_text


class InlineConfidenceEstimator(BaseConfidenceEstimator):
    name = "inline"

    def estimate(self, example, model, tokenizer):
        return float(example.get("confidence", 0.0)), {"confidence_method": "inline"}


class SelfReportConfidenceEstimator(BaseConfidenceEstimator):
    name = "self_report"

    def __init__(self, max_new_tokens: int = 64):
        self.max_new_tokens = max_new_tokens
        self._template = load_prompt("confidence_self_report")

    def estimate(self, example, model, tokenizer):
        prompt = self._template.format(
            context=example["context"],
            question=example["question"],
            answer=example["prediction"],
        )
        raw = generate_text(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            greedy=True,
        )
        conf = parse_probability_response(raw, "confidence")
        return conf, {"confidence_method": "self_report", "confidence_raw_output": raw}


class TokenEntropyConfidenceEstimator(BaseConfidenceEstimator):
    name = "token_entropy"

    def estimate(self, example, model, tokenizer):
        entropy = example.get("token_entropy_proxy")
        if entropy is None:
            conf = float(example.get("mean_token_probability", 0.0))
        else:
            conf = float(np.exp(-float(entropy)))
        conf = max(0.0, min(1.0, conf))
        return conf, {"confidence_method": "token_entropy", "entropy_proxy": entropy}


class PTrueConfidenceEstimator(BaseConfidenceEstimator):
    name = "p_true"

    def __init__(self, max_new_tokens: int = 64):
        self.max_new_tokens = max_new_tokens
        self._template = load_prompt("confidence_p_true")

    def estimate(self, example, model, tokenizer):
        prompt = self._template.format(
            question=example["question"],
            context=example["context"],
            answer=example["prediction"],
        )
        raw = generate_text(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            greedy=True,
        )
        conf = parse_probability_response(raw, "p_true")
        return conf, {"confidence_method": "p_true", "confidence_raw_output": raw}


class SelfConsistencyConfidenceEstimator(BaseConfidenceEstimator):
    name = "self_consistency"

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

    def estimate(self, example, model, tokenizer):
        from src.generation.qa import generate_answer

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

        majority = max(set(answers), key=answers.count)
        conf = answers.count(majority) / len(answers)
        return conf, {
            "confidence_method": "self_consistency",
            "samples": answers,
            "majority_answer": majority,
        }


class SemanticEntropyConfidenceEstimator(BaseConfidenceEstimator):
    """Semantic entropy via NLI-based clustering of sampled answers.

    Approximates Farquhar et al. 2024. Falls back to self-consistency if an
    NLI model is not available in the environment.
    """

    name = "semantic_entropy"

    def __init__(
        self,
        num_samples: int = 5,
        sample_temperature: float = 0.7,
        sample_top_p: float = 0.95,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
        max_new_tokens: int = 256,
    ):
        self.num_samples = num_samples
        self.sample_temperature = sample_temperature
        self.sample_top_p = sample_top_p
        self.nli_model_name = nli_model_name
        self.max_new_tokens = max_new_tokens
        self._nli: Any | None = None

    def _load_nli(self) -> Any | None:
        if self._nli is not None:
            return self._nli
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._nli = CrossEncoder(self.nli_model_name)
            return self._nli
        except Exception:  # pragma: no cover - optional dep
            return None

    @staticmethod
    def _cluster_by_entailment(answers: list[str], nli_model: Any) -> list[int]:
        """Greedy clustering: assign each answer to the first cluster that
        mutually entails it. Returns cluster ids per answer."""
        cluster_ids: list[int] = []
        representatives: list[str] = []
        for ans in answers:
            assigned = -1
            for cid, rep in enumerate(representatives):
                pairs = [(rep, ans), (ans, rep)]
                scores = nli_model.predict(pairs, show_progress_bar=False)
                labels = [int(np.argmax(s)) for s in scores]
                # CrossEncoder NLI labels: 0=contradiction, 1=entailment, 2=neutral
                if all(lbl == 1 for lbl in labels):
                    assigned = cid
                    break
            if assigned == -1:
                representatives.append(ans)
                assigned = len(representatives) - 1
            cluster_ids.append(assigned)
        return cluster_ids

    def estimate(self, example, model, tokenizer):
        from src.generation.qa import generate_answer

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
            answers.append(sample["answer"].strip())

        if not answers:
            return 0.0, {"confidence_method": "semantic_entropy", "samples": []}

        nli = self._load_nli()
        if nli is None:
            # Fallback: exact-match clustering
            clusters = {}
            for a in answers:
                clusters.setdefault(a.lower(), 0)
                clusters[a.lower()] += 1
            cluster_probs = np.array(list(clusters.values()), dtype=float)
        else:
            cluster_ids = self._cluster_by_entailment(answers, nli)
            counts = np.bincount(cluster_ids)
            cluster_probs = counts.astype(float)

        cluster_probs = cluster_probs / cluster_probs.sum()
        entropy = float(-(cluster_probs * np.log(cluster_probs + 1e-12)).sum())
        max_entropy = float(np.log(len(cluster_probs))) if len(cluster_probs) > 1 else 1.0
        # Confidence = 1 - normalised entropy
        conf = 1.0 - entropy / max_entropy if max_entropy > 0 else 1.0
        conf = max(0.0, min(1.0, conf))
        return conf, {
            "confidence_method": "semantic_entropy",
            "samples": answers,
            "semantic_entropy": entropy,
            "num_clusters": int(len(cluster_probs)),
            "nli_available": nli is not None,
        }
