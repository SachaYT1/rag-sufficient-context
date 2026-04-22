"""Passage-aware sufficient-context autorater."""

from __future__ import annotations

import json
from typing import Any

from tqdm import tqdm

from src.utils import chunk_text, generate_text, split_passages_by_token_budget


AUTORATER_PROMPT_TEMPLATE = """You are judging whether the provided evidence is sufficient to answer the question.

Question:
{question}

Evidence:
{context}

Return ONLY valid JSON:
{{
  "sufficient": true,
  "reason": "brief reason"
}}

Mark sufficient=true only if the evidence contains enough information to answer the question definitively without outside knowledge.
"""


def parse_autorater_response(raw_output: str) -> dict[str, Any]:
    """Parse autorater response into a structured dictionary."""
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw_output[start:end])
            sufficient = bool(parsed.get("sufficient", False))
            reason = str(parsed.get("reason", "")).strip()
            return {
                "sufficient": sufficient,
                "reason": reason,
                "raw_output": raw_output,
                "parsed": True,
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    lower = raw_output.lower()
    if "insufficient" in lower or "not sufficient" in lower:
        return {
            "sufficient": False,
            "reason": raw_output.strip(),
            "raw_output": raw_output,
            "parsed": False,
        }
    if "sufficient" in lower:
        return {
            "sufficient": True,
            "reason": raw_output.strip(),
            "raw_output": raw_output,
            "parsed": False,
        }
    return {
        "sufficient": False,
        "reason": raw_output.strip(),
        "raw_output": raw_output,
        "parsed": False,
    }


def _render_passage_group(passages: list[dict[str, Any]]) -> str:
    lines = []
    for passage in passages:
        title = passage.get("title", "Untitled")
        text = passage.get("text", "")
        lines.append(f"[{title}]\n{text}")
    return "\n\n".join(lines)


def rate_single_context(
    question: str,
    context: str,
    model,
    tokenizer,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Rate sufficiency for a single context string."""
    prompt = AUTORATER_PROMPT_TEMPLATE.format(question=question, context=context)
    raw_output = generate_text(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        greedy=True,
    )
    return parse_autorater_response(raw_output)


def _fallback_token_chunk_records(
    question: str,
    context: str,
    model,
    tokenizer,
    chunk_size_tokens: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Fallback path if passage metadata is missing."""
    chunks = chunk_text(context, chunk_size_tokens, tokenizer)
    records: list[dict[str, Any]] = []
    for chunk_id, chunk in enumerate(chunks):
        rating = rate_single_context(
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


def aggregate_passage_ratings(
    passage_records: list[dict[str, Any]],
    aggregation: str = "support_all_required",
    supporting_fact_titles: list[str] | None = None,
) -> bool:
    """Aggregate passage/group-level sufficiency decisions.

    - support_all_required: all gold support titles must appear among positive segments
    - or: at least one positive segment
    - and: all valid segments must be positive
    """
    if not passage_records:
        return False

    valid = [r for r in passage_records if isinstance(r.get("sufficient"), bool)]
    if not valid:
        return False

    if aggregation == "or":
        return any(r["sufficient"] for r in valid)

    if aggregation == "and":
        return all(r["sufficient"] for r in valid)

    positive_titles = set()
    for record in valid:
        if record["sufficient"]:
            positive_titles.update(record.get("segment_titles", []))

    if supporting_fact_titles:
        return set(supporting_fact_titles).issubset(positive_titles)

    positive_count = sum(r["sufficient"] for r in valid)
    return positive_count >= max(1, len(valid))


def rate_sufficiency(
    question: str,
    context: str,
    model,
    tokenizer,
    chunk_size_tokens: int = 1400,
    aggregation: str = "support_all_required",
    passages: list[dict[str, Any]] | None = None,
    supporting_fact_titles: list[str] | None = None,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Rate sufficiency using passage-aware grouping whenever possible."""
    if passages:
        groups = split_passages_by_token_budget(
            passages=passages,
            tokenizer=tokenizer,
            max_tokens_per_group=chunk_size_tokens,
        )
        passage_records: list[dict[str, Any]] = []
        for segment_id, group in enumerate(groups):
            group_text = _render_passage_group(group)
            rating = rate_single_context(
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
    model,
    tokenizer,
    chunk_size_tokens: int = 1400,
    aggregation: str = "support_all_required",
    max_new_tokens: int = 96,
) -> list[dict[str, Any]]:
    """Run passage-aware sufficiency rating for all examples."""
    results: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc="Rating context sufficiency"):
        selected_passages = ex.get("retrieved_passages")
        rating = rate_sufficiency(
            question=ex["question"],
            context=ex["context"],
            model=model,
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            aggregation=aggregation,
            passages=selected_passages,
            supporting_fact_titles=ex.get("supporting_fact_titles"),
            max_new_tokens=max_new_tokens,
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