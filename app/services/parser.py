"""Parsing utilities for model outputs.

These helpers are import-safe and contain no executable script logic.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


_FALLBACK_PAYLOAD: Dict[str, Any] = {
    "test_cases": [],
    "_parse_error": True,
}


def normalize_output(text: str) -> str:
    """Normalize raw model output before attempting JSON parsing."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    # Drop fenced code blocks while keeping inner content.
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    return cleaned


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing braces/brackets."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _find_json_fragment(text: str) -> str:
    """Find the first JSON object or array within a string."""
    if not text:
        return ""

    obj_match = re.search(r"\{[\s\S]+\}", text)
    arr_match = re.search(r"\[[\s\S]+\]", text)

    if obj_match and arr_match:
        return obj_match.group() if obj_match.start() < arr_match.start() else arr_match.group()

    if obj_match:
        return obj_match.group()

    if arr_match:
        return arr_match.group()

    return ""


def safe_parse_json(text: str) -> Tuple[Dict[str, Any], bool]:
    """Safely parse JSON with defensive cleanup.

    Returns a tuple of (payload, parse_error).
    """
    normalized = normalize_output(text)
    if not normalized:
        return dict(_FALLBACK_PAYLOAD), True

    for candidate in (normalized, _find_json_fragment(normalized)):
        if not candidate:
            continue

        cleaned = _strip_trailing_commas(candidate)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return {"test_cases": parsed}, False
            if isinstance(parsed, dict):
                return parsed, False
        except json.JSONDecodeError:
            continue

    return dict(_FALLBACK_PAYLOAD), True


def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from a model response and normalize to a dict payload."""
    payload, parse_error = safe_parse_json(text)

    if parse_error:
        return dict(_FALLBACK_PAYLOAD)

    if isinstance(payload, dict):
        if "test_cases" not in payload:
            payload = {"test_cases": payload.get("tests", [])}
        payload.setdefault("_parse_error", False)
        return payload

    return dict(_FALLBACK_PAYLOAD)


def validate_testcase_structure(payload: Any) -> Dict[str, Any]:
    """Validate and normalize testcase structure.

    Returns a dict with normalized test cases and validation errors.
    """
    errors: List[str] = []
    test_cases: List[Dict[str, Any]] = []

    if isinstance(payload, list):
        raw_cases = payload
    elif isinstance(payload, dict):
        raw_cases = payload.get("test_cases", [])
    else:
        raw_cases = []

    if not isinstance(raw_cases, list):
        errors.append("test_cases is not a list")
        raw_cases = []

    for idx, item in enumerate(raw_cases):
        if isinstance(item, dict):
            test_cases.append(item)
        else:
            errors.append(f"test_case[{idx}] is not an object")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "test_cases": test_cases,
    }
