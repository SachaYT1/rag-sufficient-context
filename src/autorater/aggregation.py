"""Aggregation strategies for per-segment autorater decisions."""

from __future__ import annotations

from typing import Any


def aggregate_passage_ratings(
    passage_records: list[dict[str, Any]],
    aggregation: str = "support_all_required",
    supporting_fact_titles: list[str] | None = None,
) -> bool:
    """Aggregate per-passage sufficiency decisions into a single bool.

    Strategies:
    - support_all_required: all gold support titles must appear among positive segments
    - or: at least one positive segment
    - and: all segments must be positive
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

    positive_titles: set[str] = set()
    for record in valid:
        if record["sufficient"]:
            positive_titles.update(record.get("segment_titles", []))

    if supporting_fact_titles:
        return set(supporting_fact_titles).issubset(positive_titles)

    return sum(r["sufficient"] for r in valid) >= max(1, len(valid))
