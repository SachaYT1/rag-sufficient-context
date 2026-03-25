"""BM25 retrieval and context construction for HotPotQA."""

import re
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.utils import truncate_text_to_tokens


def load_hotpotqa(split: str = "validation", num_examples: int = 500, seed: int = 42) -> list[dict]:
    """Load HotPotQA dataset subset.

    Returns list of dicts with keys: question, answer, supporting_facts, context.
    """
    dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed).select(range(min(num_examples, len(dataset))))

    examples = []
    for item in dataset:
        # HotPotQA context is list of (title, sentences) pairs
        passages = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            passage_text = f"{title}: {' '.join(sentences)}"
            passages.append(passage_text)

        examples.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "level": item["level"],
            "passages": passages,
        })
    return examples


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(passages: list[str]) -> BM25Okapi:
    """Build BM25 index over a list of passages."""
    tokenized = [tokenize_simple(p) for p in passages]
    return BM25Okapi(tokenized)


def retrieve_context(
    question: str,
    passages: list[str],
    bm25_index: BM25Okapi,
    top_k: int = 5,
    max_context_tokens: int = 4096,
    tokenizer=None,
) -> str:
    """Retrieve top-k passages and concatenate into context string.

    If tokenizer is provided, truncates to max_context_tokens.
    """
    query_tokens = tokenize_simple(question)
    scores = bm25_index.get_scores(query_tokens)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    # Concatenate passages
    context_parts = []
    for idx in top_indices:
        context_parts.append(passages[idx])

    context = "\n\n".join(context_parts)

    # Truncate if tokenizer provided
    if tokenizer is not None:
        context = truncate_text_to_tokens(context, max_context_tokens, tokenizer)

    return context


def build_retrieval_pipeline(examples: list[dict], top_k: int = 5, max_context_tokens: int = 4096, tokenizer=None) -> list[dict]:
    """Run retrieval for all examples.

    For each example, uses the provided distractor passages from HotPotQA
    and retrieves top-k via BM25.
    """
    results = []
    for ex in tqdm(examples, desc="Retrieving contexts"):
        passages = ex["passages"]
        bm25 = build_bm25_index(passages)
        context = retrieve_context(
            ex["question"], passages, bm25, top_k=top_k,
            max_context_tokens=max_context_tokens, tokenizer=tokenizer,
        )
        results.append({
            **ex,
            "context": context,
        })
    return results
