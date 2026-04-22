"""BM25 retrieval and retrieval diagnostics for HotPotQA distractor setting."""

from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.utils import count_tokens, truncate_text_to_tokens


def load_hotpotqa(
    split: str = "validation",
    num_examples: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load a shuffled HotPotQA distractor subset with retrieval-aware metadata."""
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split=split,
        trust_remote_code=True,
    )
    dataset = dataset.shuffle(seed=seed).select(range(min(num_examples, len(dataset))))

    examples: list[dict[str, Any]] = []
    for item in dataset:
        supporting_facts = item.get("supporting_facts", {})
        support_titles = supporting_facts.get("title", []) if supporting_facts else []
        support_sent_ids = supporting_facts.get("sent_id", []) if supporting_facts else []

        passages: list[dict[str, Any]] = []
        for idx, (title, sentences) in enumerate(
            zip(item["context"]["title"], item["context"]["sentences"])
        ):
            sentence_text = " ".join(sentences).strip()
            passage_text = f"{title}: {sentence_text}".strip()
            is_support = title in support_titles
            supporting_sentence_indices = [
                sent_id
                for support_title, sent_id in zip(support_titles, support_sent_ids)
                if support_title == title
            ]
            passages.append(
                {
                    "passage_id": idx,
                    "title": title,
                    "sentences": sentences,
                    "text": passage_text,
                    "is_supporting_fact_title": is_support,
                    "supporting_sentence_indices": supporting_sentence_indices,
                }
            )

        examples.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "level": item["level"],
                "supporting_facts": {
                    "title": support_titles,
                    "sent_id": support_sent_ids,
                },
                "supporting_fact_titles": sorted(set(support_titles)),
                "passages": passages,
            }
        )
    return examples


def tokenize_simple(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(passages: list[dict[str, Any]] | list[str]) -> BM25Okapi:
    """Build BM25 over passages."""
    normalized_passages = []
    for passage in passages:
        if isinstance(passage, dict):
            normalized_passages.append(passage["text"])
        else:
            normalized_passages.append(passage)
    tokenized = [tokenize_simple(p) for p in normalized_passages]
    return BM25Okapi(tokenized)


def _passages_to_context(passages: list[dict[str, Any]]) -> str:
    return "\n\n".join(p["text"] for p in passages)


def retrieve_context(
    question: str,
    passages: list[dict[str, Any]],
    bm25_index: BM25Okapi,
    top_k: int = 5,
    max_context_tokens: int = 4096,
    tokenizer=None,
    supporting_fact_titles: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve top-k passages and return an explainable retrieval record."""
    query_tokens = tokenize_simple(question)
    scores = bm25_index.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: float(scores[i]),
        reverse=True,
    )
    top_indices = ranked_indices[:top_k]
    selected_passages = [passages[idx] for idx in top_indices]
    context_pre_trunc = _passages_to_context(selected_passages)

    if tokenizer is not None:
        context, trunc_meta = truncate_text_to_tokens(
            context_pre_trunc,
            max_tokens=max_context_tokens,
            tokenizer=tokenizer,
            return_metadata=True,
        )
    else:
        context = context_pre_trunc
        approx_tokens = len(context.split())
        trunc_meta = {
            "context_tokens_before_truncation": approx_tokens,
            "context_tokens_after_truncation": approx_tokens,
            "was_truncated": False,
        }

    selected_titles = [p["title"] for p in selected_passages]
    support_title_set = set(supporting_fact_titles or [])
    support_titles_retrieved_pre_trunc = sorted(set(selected_titles) & support_title_set)

    if tokenizer is not None and trunc_meta["was_truncated"]:
        surviving_titles: list[str] = []
        remaining_context = context
        for passage in selected_passages:
            if passage["text"] in remaining_context:
                surviving_titles.append(passage["title"])
        support_titles_retrieved_post_trunc = sorted(set(surviving_titles) & support_title_set)
    else:
        support_titles_retrieved_post_trunc = support_titles_retrieved_pre_trunc

    support_recall_pre = (
        len(support_titles_retrieved_pre_trunc) / len(support_title_set)
        if support_title_set
        else None
    )
    support_recall_post = (
        len(support_titles_retrieved_post_trunc) / len(support_title_set)
        if support_title_set
        else None
    )

    retrieval_record = {
        "context": context,
        "retrieved_indices": top_indices,
        "retrieved_titles": selected_titles,
        "retrieved_passages": selected_passages,
        "bm25_scores": [float(scores[idx]) for idx in top_indices],
        "bm25_ranked_indices": ranked_indices,
        "context_tokens_before_truncation": trunc_meta["context_tokens_before_truncation"],
        "context_tokens_after_truncation": trunc_meta["context_tokens_after_truncation"],
        "was_truncated": trunc_meta["was_truncated"],
        "support_titles_retrieved_pre_truncation": support_titles_retrieved_pre_trunc,
        "support_titles_retrieved_post_truncation": support_titles_retrieved_post_trunc,
        "support_title_recall_pre_truncation": support_recall_pre,
        "support_title_recall_post_truncation": support_recall_post,
        "full_support_coverage_pre_truncation": (
            set(support_titles_retrieved_pre_trunc) == support_title_set
            if support_title_set
            else None
        ),
        "full_support_coverage_post_truncation": (
            set(support_titles_retrieved_post_trunc) == support_title_set
            if support_title_set
            else None
        ),
        "lost_support_after_truncation": (
            len(set(support_titles_retrieved_pre_trunc) - set(support_titles_retrieved_post_trunc)) > 0
            if support_title_set
            else False
        ),
        "num_selected_passages": len(selected_passages),
        "top1_bm25_score": float(scores[top_indices[0]]) if top_indices else None,
        "mean_topk_bm25_score": (
            sum(float(scores[idx]) for idx in top_indices) / len(top_indices)
            if top_indices
            else None
        ),
    }
    return retrieval_record


def build_retrieval_pipeline(
    examples: list[dict[str, Any]],
    top_k: int = 5,
    max_context_tokens: int = 4096,
    tokenizer=None,
) -> list[dict[str, Any]]:
    """Run per-question BM25 re-ranking and attach retrieval diagnostics."""
    results: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc="Retrieving contexts"):
        passages = ex["passages"]
        bm25 = build_bm25_index(passages)
        retrieval = retrieve_context(
            question=ex["question"],
            passages=passages,
            bm25_index=bm25,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
            tokenizer=tokenizer,
            supporting_fact_titles=ex.get("supporting_fact_titles"),
        )
        results.append({**ex, **retrieval})
    return results


def summarize_retrieval_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval metrics for plotting and diagnostics."""
    total = len(examples) or 1

    pre_recalls = [
        ex["support_title_recall_pre_truncation"]
        for ex in examples
        if ex.get("support_title_recall_pre_truncation") is not None
    ]
    post_recalls = [
        ex["support_title_recall_post_truncation"]
        for ex in examples
        if ex.get("support_title_recall_post_truncation") is not None
    ]
    full_pre = [
        ex["full_support_coverage_pre_truncation"]
        for ex in examples
        if ex.get("full_support_coverage_pre_truncation") is not None
    ]
    full_post = [
        ex["full_support_coverage_post_truncation"]
        for ex in examples
        if ex.get("full_support_coverage_post_truncation") is not None
    ]
    top1_scores = [ex["top1_bm25_score"] for ex in examples if ex.get("top1_bm25_score") is not None]
    mean_topk_scores = [
        ex["mean_topk_bm25_score"] for ex in examples if ex.get("mean_topk_bm25_score") is not None
    ]

    return {
        "num_examples": len(examples),
        "truncation_rate": sum(bool(ex.get("was_truncated", False)) for ex in examples) / total,
        "mean_context_tokens_before_truncation": (
            sum(int(ex.get("context_tokens_before_truncation", 0)) for ex in examples) / total
        ),
        "mean_context_tokens_after_truncation": (
            sum(int(ex.get("context_tokens_after_truncation", 0)) for ex in examples) / total
        ),
        "mean_selected_passages": (
            sum(int(ex.get("num_selected_passages", 0)) for ex in examples) / total
        ),
        "support_title_recall_at_k_pre_truncation": (
            sum(pre_recalls) / len(pre_recalls) if pre_recalls else None
        ),
        "support_title_recall_at_k_post_truncation": (
            sum(post_recalls) / len(post_recalls) if post_recalls else None
        ),
        "full_support_coverage_at_k_pre_truncation": (
            sum(float(v) for v in full_pre) / len(full_pre) if full_pre else None
        ),
        "full_support_coverage_at_k_post_truncation": (
            sum(float(v) for v in full_post) / len(full_post) if full_post else None
        ),
        "lost_support_after_truncation_rate": (
            sum(bool(ex.get("lost_support_after_truncation", False)) for ex in examples) / total
        ),
        "mean_top1_bm25_score": sum(top1_scores) / len(top1_scores) if top1_scores else None,
        "mean_topk_bm25_score": (
            sum(mean_topk_scores) / len(mean_topk_scores) if mean_topk_scores else None
        ),
    }