"""Confidence estimation: P(Correct) self-reported probability."""

import json
from tqdm import tqdm


CONFIDENCE_PROMPT_TEMPLATE = """You previously answered the following question based on a context.

Context:
{context}

Question: {question}

Your answer was: {answer}

How confident are you that this answer is correct? Consider:
- Does the context directly support this answer?
- Are there any contradictions or ambiguities?
- Could there be other valid answers?

Respond with ONLY a JSON object:
{{"confidence": 0.XX}}

Where 0.XX is a number between 0.0 (not confident at all) and 1.0 (completely certain).

JSON response:"""


def parse_confidence_response(raw_output: str) -> float:
    """Parse confidence value from LLM response."""
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw_output[start:end])
            conf = float(parsed.get("confidence", 0.0))
            return max(0.0, min(1.0, conf))
    except (json.JSONDecodeError, ValueError):
        pass
    return 0.0


def estimate_confidence_separate(
    question: str,
    context: str,
    answer: str,
    model,
    tokenizer,
    max_new_tokens: int = 64,
) -> float:
    """Estimate confidence via a separate follow-up prompt.

    This is an alternative to the inline confidence from generation.py.
    Useful for re-calibrating confidence estimates.
    """
    prompt = CONFIDENCE_PROMPT_TEMPLATE.format(
        context=context, question=question, answer=answer,
    )

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    import torch
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return parse_confidence_response(raw_output)


def estimate_confidence_batch(
    examples: list[dict],
    model,
    tokenizer,
    use_inline: bool = True,
) -> list[dict]:
    """Estimate confidence for all examples.

    If use_inline=True, uses the confidence already extracted during generation.
    If use_inline=False, runs a separate confidence estimation prompt.
    """
    results = []
    for ex in tqdm(examples, desc="Estimating confidence"):
        if use_inline and "confidence" in ex:
            conf = ex["confidence"]
        else:
            conf = estimate_confidence_separate(
                ex["question"], ex["context"], ex["prediction"],
                model, tokenizer,
            )

        results.append({**ex, "confidence": conf})
    return results
