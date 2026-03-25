"""Sufficient Context Autorater: binary classifier for (question, context) pairs."""

import json
from tqdm import tqdm

from src.utils import chunk_text, generate_text


AUTORATER_PROMPT_TEMPLATE = """You are evaluating whether the provided context contains enough information to answer the given question.

Context:
{context}

Question: {question}

Does the context contain sufficient information to definitively answer this question?
Consider:
- Are all necessary facts present in the context?
- Is there any ambiguity that cannot be resolved from the context alone?
- Would answering require knowledge not present in the context?

Respond with ONLY a JSON object:
{{"sufficient": true}} or {{"sufficient": false}}

JSON response:"""


def parse_autorater_response(raw_output: str) -> bool | None:
    """Parse autorater response to extract sufficiency label."""
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw_output[start:end])
            return bool(parsed.get("sufficient", False))
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: look for keywords
    lower = raw_output.lower()
    if "sufficient" in lower and "insufficient" not in lower:
        return True
    if "insufficient" in lower or "not sufficient" in lower:
        return False

    return None


def rate_single_chunk(
    question: str,
    chunk: str,
    model,
    tokenizer,
    max_new_tokens: int = 64,
) -> bool | None:
    """Rate sufficiency for a single (question, chunk) pair."""
    prompt = AUTORATER_PROMPT_TEMPLATE.format(context=chunk, question=question)
    raw_output = generate_text(prompt, model, tokenizer, max_new_tokens=max_new_tokens, greedy=True)
    return parse_autorater_response(raw_output)


def rate_sufficiency(
    question: str,
    context: str,
    model,
    tokenizer,
    chunk_size_tokens: int = 1400,
    aggregation: str = "or",
) -> dict:
    """Rate context sufficiency with chunking and aggregation.

    Args:
        question: the question
        context: the full retrieved context
        model: the LLM model
        tokenizer: the tokenizer
        chunk_size_tokens: tokens per chunk
        aggregation: 'or' (sufficient if any chunk is) or 'and' (all must be)

    Returns:
        dict with 'sufficient' (bool), 'chunk_results' (list), 'num_chunks' (int)
    """
    chunks = chunk_text(context, chunk_size_tokens, tokenizer)
    if not chunks:
        return {"sufficient": False, "chunk_results": [], "num_chunks": 0}

    chunk_results = []
    for chunk in chunks:
        result = rate_single_chunk(question, chunk, model, tokenizer)
        chunk_results.append(result)

    # Aggregate
    valid_results = [r for r in chunk_results if r is not None]
    if not valid_results:
        sufficient = False
    elif aggregation == "or":
        sufficient = any(valid_results)
    else:  # "and"
        sufficient = all(valid_results)

    return {
        "sufficient": sufficient,
        "chunk_results": chunk_results,
        "num_chunks": len(chunks),
    }


def rate_all_examples(
    examples: list[dict],
    model,
    tokenizer,
    chunk_size_tokens: int = 1400,
    aggregation: str = "or",
) -> list[dict]:
    """Rate sufficiency for all examples."""
    results = []
    for ex in tqdm(examples, desc="Rating context sufficiency"):
        rating = rate_sufficiency(
            ex["question"], ex["context"], model, tokenizer,
            chunk_size_tokens=chunk_size_tokens, aggregation=aggregation,
        )
        results.append({
            **ex,
            "sufficient": rating["sufficient"],
            "num_chunks": rating["num_chunks"],
        })
    return results
