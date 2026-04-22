"""Passage-aware sufficiency rating pipeline."""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.autorater.aggregation import aggregate_passage_ratings
from src.autorater.strategies import BaseAutorater, PromptAutorater
from src.utils import chunk_text, split_passages_by_token_budget


def _render_passage_group(passages: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{p.get('title', 'Untitled')}]\n{p.get('text', '')}" for p in passages)


def _fallback_token_chunk_records(
    autorater: BaseAutorater,
    question: str,
    context: str,
    model: Any,
    tokenizer: Any,
    chunk_size_tokens: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    chunks = chunk_text(context, chunk_size_tokens, tokenizer)
    records: list[dict[str, Any]] = []
    for chunk_id, chunk in enumerate(chunks):
        rating = autorater.rate(
            question=question,
            context=chunk,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        records.append(
            {
                "segment_id": chunk_id,
                "segment_type": "token_chunk",
                "segment_titles": [],
                "segment_text": chunk,
                **rating,
            }
        )
    return records


def rate_single_context(
    question: str,
    context: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Backward-compatible single-prompt rating."""
    return PromptAutorater(prompt_name="autorater_basic", name="basic").rate(
        question=question,
        context=context,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )


def rate_sufficiency(
    question: str,
    context: str,
    model: Any,
    tokenizer: Any,
    chunk_size_tokens: int = 1400,
    aggregation: str = "support_all_required",
    passages: list[dict[str, Any]] | None = None,
    supporting_fact_titles: list[str] | None = None,
    max_new_tokens: int = 96,
    autorater: BaseAutorater | None = None,
) -> dict[str, Any]:
    """Passage-aware sufficiency rating."""
    active = autorater or PromptAutorater(prompt_name="autorater_basic", name="basic")

    if passages:
        groups = split_passages_by_token_budget(
            passages=passages,
            tokenizer=tokenizer,
            max_tokens_per_group=chunk_size_tokens,
        )
        passage_records: list[dict[str, Any]] = []
        for segment_id, group in enumerate(groups):
            group_text = _render_passage_group(group)
            rating = active.rate(
                question=question,
                context=group_text,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
            passage_records.append(
                {
                    "segment_id": segment_id,
                    "segment_type": "passage_group",
                    "segment_titles": [p.get("title") for p in group],
                    "segment_text": group_text,
                    **rating,
                }
            )
    else:
        passage_records = _fallback_token_chunk_records(
            autorater=active,
            question=question,
            context=context,
            model=model,
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            max_new_tokens=max_new_tokens,
        )

    sufficient = aggregate_passage_ratings(
        passage_records=passage_records,
        aggregation=aggregation,
        supporting_fact_titles=supporting_fact_titles,
    )
    positive_titles = sorted(
        {
            title
            for record in passage_records
            if record.get("sufficient")
            for title in record.get("segment_titles", [])
        }
    )

    return {
        "sufficient": sufficient,
        "passage_ratings": passage_records,
        "num_chunks": len(passage_records),
        "positive_chunk_ratio": (
            sum(bool(r.get("sufficient")) for r in passage_records) / len(passage_records)
            if passage_records
            else 0.0
        ),
        "positive_passage_titles": positive_titles,
    }


def rate_all_examples(
    examples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    chunk_size_tokens: int = 1400,
    aggregation: str = "support_all_required",
    max_new_tokens: int = 96,
    autorater: BaseAutorater | None = None,
) -> list[dict[str, Any]]:
    active = autorater or PromptAutorater(prompt_name="autorater_basic", name="basic")
    results: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc=f"Rating sufficiency ({active.name})"):
        rating = rate_sufficiency(
            question=ex["question"],
            context=ex["context"],
            model=model,
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            aggregation=aggregation,
            passages=ex.get("retrieved_passages"),
            supporting_fact_titles=ex.get("supporting_fact_titles"),
            max_new_tokens=max_new_tokens,
            autorater=active,
        )
        results.append(
            {
                **ex,
                "sufficient": rating["sufficient"],
                "num_chunks": rating["num_chunks"],
                "passage_ratings": rating["passage_ratings"],
                "positive_chunk_ratio": rating["positive_chunk_ratio"],
                "positive_passage_titles": rating["positive_passage_titles"],
            }
        )
    return results
