"""HotPotQA distractor loader."""

from __future__ import annotations

from typing import Any

from datasets import load_dataset


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
