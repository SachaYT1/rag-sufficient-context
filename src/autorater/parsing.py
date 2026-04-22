"""Robust parser for autorater JSON responses."""

from __future__ import annotations

import json
from typing import Any


def parse_autorater_response(raw_output: str) -> dict[str, Any]:
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw_output[start:end])
            return {
                "sufficient": bool(parsed.get("sufficient", False)),
                "reason": str(parsed.get("reason", "")).strip(),
                "raw_output": raw_output,
                "parsed": True,
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    lower = raw_output.lower()
    if "insufficient" in lower or "not sufficient" in lower:
        return {"sufficient": False, "reason": raw_output.strip(), "raw_output": raw_output, "parsed": False}
    if "sufficient" in lower:
        return {"sufficient": True, "reason": raw_output.strip(), "raw_output": raw_output, "parsed": False}
    return {"sufficient": False, "reason": raw_output.strip(), "raw_output": raw_output, "parsed": False}
