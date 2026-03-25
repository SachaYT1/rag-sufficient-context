"""Evaluation: EM/F1 metrics, abstention detection, output categorization."""

import re
import string
from collections import Counter


# Abstention patterns
ABSTENTION_PATTERNS = [
    r"i don'?t know",
    r"cannot be determined",
    r"not enough information",
    r"insufficient information",
    r"cannot answer",
    r"unable to answer",
    r"no answer",
    r"not possible to determine",
    r"the context does not",
    r"the provided context does not",
    r"cannot be answered",
    r"i'm not sure",
    r"there is not enough",
]


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison (lowercase, remove articles/punctuation)."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def exact_match(prediction: str, gold: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def is_abstention(text: str) -> bool:
    """Check if the response is an abstention."""
    text_lower = text.lower().strip()
    for pattern in ABSTENTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def categorize_response(prediction: str, gold: str, f1_threshold: float = 0.5) -> str:
    """Categorize response as 'correct', 'abstain', or 'hallucinate'.

    Args:
        prediction: model's predicted answer
        gold: ground truth answer
        f1_threshold: minimum F1 to consider correct

    Returns:
        One of 'correct', 'abstain', 'hallucinate'
    """
    # NOTE: Abstention is checked first by design. If a response contains both
    # an abstention phrase and the correct answer (e.g., "It might be Paris but
    # I don't know"), we treat it as abstention since the model is signaling
    # uncertainty rather than committing to an answer.
    if is_abstention(prediction):
        return "abstain"

    em = exact_match(prediction, gold)
    f1 = f1_score(prediction, gold)

    if em > 0 or f1 >= f1_threshold:
        return "correct"

    return "hallucinate"


def evaluate_all(examples: list[dict], f1_threshold: float = 0.5) -> dict:
    """Evaluate all examples and return metrics.

    Each example must have 'prediction' and 'answer' keys.

    Returns dict with per-example results and aggregate metrics.
    """
    categories = {"correct": 0, "abstain": 0, "hallucinate": 0}
    em_scores = []
    f1_scores = []
    per_example = []

    for ex in examples:
        pred = ex["prediction"]
        gold = ex["answer"]

        category = categorize_response(pred, gold, f1_threshold)
        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)

        categories[category] += 1
        em_scores.append(em)
        f1_scores.append(f1)

        per_example.append({
            "id": ex.get("id", ""),
            "question": ex["question"],
            "prediction": pred,
            "answer": gold,
            "category": category,
            "em": em,
            "f1": f1,
        })

    total = len(examples)
    metrics = {
        "total": total,
        "correct": categories["correct"],
        "abstain": categories["abstain"],
        "hallucinate": categories["hallucinate"],
        "correct_rate": categories["correct"] / total if total > 0 else 0,
        "abstain_rate": categories["abstain"] / total if total > 0 else 0,
        "hallucinate_rate": categories["hallucinate"] / total if total > 0 else 0,
        "mean_em": sum(em_scores) / total if total > 0 else 0,
        "mean_f1": sum(f1_scores) / total if total > 0 else 0,
    }

    return {"metrics": metrics, "per_example": per_example}
