"""Autorater aggregation strategies."""

from __future__ import annotations

from src.autorater.aggregation import aggregate_passage_ratings


def _rec(sufficient: bool, titles: list[str]) -> dict:
    return {"sufficient": sufficient, "segment_titles": titles}


def test_or_aggregation() -> None:
    records = [_rec(False, ["a"]), _rec(True, ["b"])]
    assert aggregate_passage_ratings(records, aggregation="or")


def test_and_aggregation() -> None:
    records = [_rec(True, ["a"]), _rec(False, ["b"])]
    assert not aggregate_passage_ratings(records, aggregation="and")


def test_support_all_required_happy_path() -> None:
    records = [_rec(True, ["title_a"]), _rec(True, ["title_b"])]
    assert aggregate_passage_ratings(
        records,
        aggregation="support_all_required",
        supporting_fact_titles=["title_a", "title_b"],
    )


def test_support_all_required_missing_title() -> None:
    records = [_rec(True, ["title_a"])]
    assert not aggregate_passage_ratings(
        records,
        aggregation="support_all_required",
        supporting_fact_titles=["title_a", "title_b"],
    )


def test_empty_records() -> None:
    assert not aggregate_passage_ratings([], aggregation="or")
