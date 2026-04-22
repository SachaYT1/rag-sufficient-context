"""High-level retrieval pipeline with diagnostics and metric aggregation."""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.utils import truncate_text_to_tokens


def _passages_to_context(passages: list[dict[str, Any]]) -> str:
    return "\n\n".join(p["text"] for p in passages)


def retrieve_context(
    question: str,
    passages: list[dict[str, Any]],
    bm25_index: Any = None,  # kept for backward compatibility
    top_k: int = 5,
    max_context_tokens: int = 4096,
    tokenizer: Any = None,
    supporting_fact_titles: list[str] | None = None,
    retriever: BaseRetriever | None = None,
) -> dict[str, Any]:
    """Retrieve top-k passages and return an explainable retrieval record."""
    active_retriever = retriever or BM25Retriever()

    if bm25_index is not None and active_retriever.name == "bm25":
        from src.retrieval.bm25 import tokenize_simple

        scores = [float(s) for s in bm25_index.get_scores(tokenize_simple(question))]
    else:
        scores = active_retriever.score(question, passages)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
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
        for passage in selected_passages:
            if passage["text"] in context:
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

    return {
        "context": context,
        "retrieved_indices": top_indices,
        "retrieved_titles": selected_titles,
        "retrieved_passages": selected_passages,
        "bm25_scores": [float(scores[idx]) for idx in top_indices],
        "retriever_scores": [float(scores[idx]) for idx in top_indices],
        "retriever_name": active_retriever.name,
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


def build_retrieval_pipeline(
    examples: list[dict[str, Any]],
    top_k: int = 5,
    max_context_tokens: int = 4096,
    tokenizer: Any = None,
    retriever: BaseRetriever | None = None,
) -> list[dict[str, Any]]:
    """Run per-question retrieval and attach diagnostics."""
    active_retriever = retriever or BM25Retriever()
    results: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc=f"Retrieving ({active_retriever.name})"):
        passages = ex["passages"]
        retrieval = retrieve_context(
            question=ex["question"],
            passages=passages,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
            tokenizer=tokenizer,
            supporting_fact_titles=ex.get("supporting_fact_titles"),
            retriever=active_retriever,
        )
        results.append({**ex, **retrieval})
    return results


def summarize_retrieval_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(examples) or 1

    def _avg(key: str) -> float | None:
        values = [ex[key] for ex in examples if ex.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "num_examples": len(examples),
        "truncation_rate": sum(bool(ex.get("was_truncated", False)) for ex in examples) / total,
        "mean_context_tokens_before_truncation": sum(
            int(ex.get("context_tokens_before_truncation", 0)) for ex in examples
        ) / total,
        "mean_context_tokens_after_truncation": sum(
            int(ex.get("context_tokens_after_truncation", 0)) for ex in examples
        ) / total,
        "mean_selected_passages": sum(
            int(ex.get("num_selected_passages", 0)) for ex in examples
        ) / total,
        "support_title_recall_at_k_pre_truncation": _avg("support_title_recall_pre_truncation"),
        "support_title_recall_at_k_post_truncation": _avg("support_title_recall_post_truncation"),
        "full_support_coverage_at_k_pre_truncation": (
            sum(float(v) for v in [ex.get("full_support_coverage_pre_truncation") for ex in examples] if v is not None)
            / max(1, sum(ex.get("full_support_coverage_pre_truncation") is not None for ex in examples))
        ),
        "full_support_coverage_at_k_post_truncation": (
            sum(float(v) for v in [ex.get("full_support_coverage_post_truncation") for ex in examples] if v is not None)
            / max(1, sum(ex.get("full_support_coverage_post_truncation") is not None for ex in examples))
        ),
        "lost_support_after_truncation_rate": sum(
            bool(ex.get("lost_support_after_truncation", False)) for ex in examples
        ) / total,
        "mean_top1_bm25_score": _avg("top1_bm25_score"),
        "mean_topk_bm25_score": _avg("mean_topk_bm25_score"),
    }
