"""NQ-Open loader.

NQ-Open does not ship with curated distractor passages, so we retrieve
Wikipedia passages on-the-fly with BM25 over a small local corpus. To keep
the default environment lightweight, we support two modes:

1. Provide ``passages_per_example`` pre-assembled passages per question via a
   local JSONL (``passages_jsonl``), each row: ``{id, question, answer, passages: [{title, text}]}``.
2. Otherwise fall back to using the ``nq_open`` huggingface dataset and
   attach an empty passage list; downstream retrievers must then bring their
   own corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _unify(item_id: str, question: str, answers: list[str], passages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": str(item_id),
        "question": question,
        "answer": answers[0] if answers else "",
        "answers": answers,
        "type": "nq_open",
        "level": "unknown",
        "supporting_facts": {"title": [], "sent_id": []},
        "supporting_fact_titles": [],
        "passages": passages,
    }


def load_nq_open(
    split: str = "validation",
    num_examples: int = 300,
    seed: int = 42,
    passages_jsonl: str | None = None,
) -> list[dict[str, Any]]:
    """Load NQ-Open with optional precomputed passages."""
    passage_index: dict[str, list[dict[str, Any]]] = {}
    if passages_jsonl:
        path = Path(passages_jsonl)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    passage_index[row["question"].strip().lower()] = row.get("passages", [])

    dataset = load_dataset("nq_open", split=split)
    dataset = dataset.shuffle(seed=seed).select(range(min(num_examples, len(dataset))))

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        q = item["question"]
        answers = item.get("answer") or []
        passages = passage_index.get(q.strip().lower(), [])
        out.append(_unify(item_id=f"nq-{idx}", question=q, answers=answers, passages=passages))
    return out
