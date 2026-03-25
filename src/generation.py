"""LLM generation: prompting LLaMA-3.1-8B-Instruct for QA."""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.utils import generate_text


QA_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided context.
If the context does not contain enough information to answer the question, say "I don't know".

Context:
{context}

Question: {question}

Provide your answer in the following JSON format:
{{"answer": "your answer here", "confidence": 0.0}}

Where confidence is a number between 0.0 and 1.0 representing how confident you are in your answer.
If you don't know the answer, set confidence to 0.0 and answer to "I don't know".

JSON response:"""


def load_model(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "auto"):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def format_prompt(question: str, context: str) -> str:
    """Format the QA prompt."""
    return QA_PROMPT_TEMPLATE.format(context=context, question=question)


def parse_llm_response(raw_output: str) -> dict:
    """Parse JSON response from LLM output.

    Returns dict with 'answer' and 'confidence' keys.
    Falls back to raw text if JSON parsing fails.
    """
    # Try to find JSON in the output
    try:
        # Look for JSON-like pattern
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            json_str = raw_output[start:end]
            parsed = json.loads(json_str)
            conf = float(parsed.get("confidence", 0.0))
            return {
                "answer": str(parsed.get("answer", "")).strip(),
                "confidence": max(0.0, min(1.0, conf)),
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: use raw text as answer with low confidence
    return {
        "answer": raw_output.strip(),
        "confidence": 0.1,
    }


def generate_answer(
    question: str,
    context: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> dict:
    """Generate answer for a single question.

    Returns dict with 'answer', 'confidence', 'raw_output'.
    """
    prompt = format_prompt(question, context)
    raw_output = generate_text(
        prompt, model, tokenizer,
        max_new_tokens=max_new_tokens, greedy=(temperature == 0.0),
    )

    parsed = parse_llm_response(raw_output)
    parsed["raw_output"] = raw_output

    return parsed


def generate_answers_batch(
    examples: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> list[dict]:
    """Generate answers for all examples."""
    results = []
    for ex in tqdm(examples, desc="Generating answers"):
        answer_data = generate_answer(
            ex["question"], ex["context"], model, tokenizer,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        results.append({
            **ex,
            "prediction": answer_data["answer"],
            "confidence": answer_data["confidence"],
            "raw_output": answer_data["raw_output"],
        })
    return results
