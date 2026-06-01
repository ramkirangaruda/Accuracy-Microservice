import os
import json
import time
import re

from dotenv import load_dotenv
from openai import OpenAI

from app.services.heuristics import evaluate_heuristics
from app.services.scorer import get_quality_grade
from app.core.logger import logger

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

JUDGE_MODEL = "llama-3.1-8b-instant"

JUDGE_MAX_RETRIES = 3
JUDGE_BACKOFF_SECONDS = 2

MAX_GENERATED_CHARS = 3000
MAX_JUDGE_TOKENS = 600
JUDGE_TIMEOUT_SECONDS = 20

DIMENSION_WEIGHTS = {
    "technical_correctness": 0.30,
    "automation_ready":      0.25,
    "assertions_testable":   0.25,
    "oracle_correct":        0.12,
    "edge_case_realism":     0.08,
}

CRITICAL_DIMENSIONS = [
    "technical_correctness",
    "automation_ready",
    "assertions_testable",
]

CRITICAL_FLOOR_THRESHOLD = 3
CRITICAL_FLOOR_CAP = 45

# ---------------------------------------------------
# PROMPT
# ---------------------------------------------------

SCORING_ANCHORS = """
CRITICAL DISAMBIGUATION — READ BEFORE SCORING:

oracle_correct measures: are the EXPECTED VALUES logically correct?
  - A test case stating "Expected Status: 200" on a successful PUT/POST/PATCH
    is a CORRECT oracle. Score this 7-8, not 0.
  - oracle_correct is NOT about whether assertions are explicit enough.
    That is assertions_testable's job. Do NOT double-penalize.
  - Score 0 ONLY if the expected value is factually wrong
    (e.g., "expect 200 on an unauthorized request").

automation_ready measures: can a tool execute this without human judgment?
  10 = runnable code (pytest, curl script, etc.)
   6 = structured spec (verb + endpoint + payload + expected status)
       that a tool could consume directly — NOT manual-only
   3 = partial structure, some fields missing
   0 = pure prose description only ("test that login works")

  A test with PUT /api/profile + JSON payload + Expected Status: 200
  is a STRUCTURED SPEC. Score it 5-6, NOT 0.

assertions_testable measures: how machine-verifiable are the assertions?
  10 = exact status code + exact response body field + exact value
   7 = exact status code + field presence check
   4 = exact status code only ("Expected Status: 200")
   2 = vague assertion present but status code also stated
   0 = no assertion whatsoever — not even a status code

  "Expected Status: 200" IS an assertion. Score minimum 4.
  A vague phrase alongside a status code is band 2-3, not 0.

technical_correctness:
  10 = correct HTTP verb, endpoint, payload, expected status, all assertions explicit
   7 = correct verb+endpoint, assertions present but one is vague
   4 = endpoint present, assertions vague ("verify it works")
   1 = no endpoint, no assertions, prose description only
   0 = factually wrong (wrong verb, wrong status code expectation)

oracle_correct:
  10 = expected values are correct per HTTP spec and feature logic
   5 = expected values plausible but not verified against spec
   0 = expected values are wrong (e.g., expects 200 on auth failure)

edge_case_realism:
  10 = covers invalid input, boundary, auth failure, and error state
   7 = covers 2-3 realistic error cases
   4 = covers one edge case
   0 = happy path only

confidence:
  0-10: how confident you are in this evaluation given the information provided
"""

JUDGE_SCHEMA = """
Return ONLY valid JSON matching this exact schema.
Every score is an integer 0-10. No exceptions.
Do not include any text outside the JSON object.

{
    "dimensions": {
        "technical_correctness": {
            "score": 0,
            "reason": "one sentence max"
        },
        "automation_ready": {
            "score": 0,
            "reason": "one sentence max"
        },
        "assertions_testable": {
            "score": 0,
            "reason": "one sentence max"
        },
        "oracle_correct": {
            "score": 0,
            "reason": "one sentence max"
        },
        "edge_case_realism": {
            "score": 0,
            "reason": "one sentence max"
        }
    },
    "summary": "",
    "strengths": [],
    "weaknesses": [],
    "critical_failures": [],
    "suggested_tags": [],
    "confidence": 0
}
"""

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------


def safe_json_extract(text):

    if not text:
        return None, "Empty judge response"

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = (
            cleaned
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    candidates = [cleaned]

    object_match = re.search(r"\{[\s\S]+\}", cleaned)

    if object_match:
        candidates.append(object_match.group())

    for candidate in candidates:

        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

        try:

            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed, None

        except Exception:
            continue

    return None, "Malformed JSON"


def is_rate_limit_error(exc: Exception) -> bool:

    msg = str(exc).lower()

    return any(
        token in msg
        for token in [
            "rate limit",
            "429",
            "quota",
            "too many requests",
        ]
    )


def is_timeout_error(exc: Exception) -> bool:

    msg = str(exc).lower()

    return any(
        token in msg
        for token in [
            "timeout",
            "timed out",
            "read timeout",
        ]
    )


def _clamp_int(value, low=0, high=10) -> int:

    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return low

    return max(low, min(high, numeric))


def _normalize_dimension(value: dict, label: str) -> dict:

    if not isinstance(value, dict):
        return {
            "score": 0,
            "reason": f"{label} missing"
        }

    score = _clamp_int(value.get("score"), 0, 10)
    reason = value.get("reason")

    if not isinstance(reason, str) or not reason.strip():
        reason = f"{label} reason missing"

    return {
        "score": score,
        "reason": reason.strip()
    }


def apply_consistency_corrections(
    dims: dict,
    generated_output: str,
    heuristics: dict
) -> tuple[dict, list[str]]:
    """
    Rule-based corrections for known LLM calibration failures.
    Fire when heuristic evidence contradicts LLM dimension scores.
    These corrections only ever LIFT scores that are provably wrong,
    never inflate scores beyond what evidence supports.
    """
    corrections = []
    text_lower = generated_output.lower()

    # ---------------------------------------------------
    # Rule 1: structured spec cannot be scored as manual-only
    # If endpoint + payload + any status code all present,
    # automation_ready cannot be 0 or 1.
    # ---------------------------------------------------
    has_status_code = bool(
        re.search(r"(expected status|status code)[:\s]+\d{3}", text_lower)
        or re.search(r"\b(200|201|204|400|401|403|404|500)\b", text_lower)
    )

    if (
        heuristics.get("has_endpoint", False)
        and heuristics.get("has_payload", False)
        and has_status_code
        and dims.get("automation_ready", {}).get("score", 10) <= 1
    ):
        dims["automation_ready"]["score"] = 5
        dims["automation_ready"]["reason"] += (
            " [corrected: structured spec present — endpoint + payload +"
            " expected status — not manual-only]"
        )
        corrections.append(
            "automation_ready lifted to 5: structured spec was incorrectly"
            " scored as manual-only (0-1)"
        )

    # ---------------------------------------------------
    # Rule 2: explicit status code is a real assertion.
    # assertions_testable cannot be 0 if a status code is stated.
    # ---------------------------------------------------
    explicit_status = re.search(
        r"(expected status|status code)[:\s]+\d{3}",
        text_lower
    )

    if (
        explicit_status
        and dims.get("assertions_testable", {}).get("score", 10) == 0
    ):
        dims["assertions_testable"]["score"] = 3
        dims["assertions_testable"]["reason"] += (
            " [corrected: explicit status code is a machine-verifiable"
            " assertion; minimum score is 3]"
        )
        corrections.append(
            "assertions_testable lifted to 3: explicit status code was"
            " scored as no assertion (0)"
        )

    # ---------------------------------------------------
    # Rule 3: HTTP-correct verb + status = valid oracle.
    # oracle_correct cannot be 0 or 1 when the combination is right.
    # ---------------------------------------------------
    correct_verb_status = (
        re.search(r"\b(put|post|patch)\b", text_lower)
        and re.search(r"\b(200|201|204)\b", text_lower)
    ) or (
        re.search(r"\bget\b", text_lower)
        and re.search(r"\b200\b", text_lower)
    ) or (
        re.search(r"\bdelete\b", text_lower)
        and re.search(r"\b(200|204)\b", text_lower)
    )

    if (
        correct_verb_status
        and dims.get("oracle_correct", {}).get("score", 10) < 5
    ):
        dims["oracle_correct"]["score"] = 7
        dims["oracle_correct"]["reason"] += (
            " [corrected: HTTP-correct verb+status combination is a valid"
            " oracle per spec]"
        )
        corrections.append(
            "oracle_correct lifted to 7: HTTP-correct verb+status was"
            " scored as invalid oracle (0-1)"
        )

    return dims, corrections


def normalize_score(parsed: dict) -> tuple[float, dict]:
    """Return (base_score_0_to_100, dimension_scores_dict)."""

    dims = parsed.get("dimensions", {})
    dimension_scores = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for dim, weight in DIMENSION_WEIGHTS.items():
        raw = dims.get(dim, {})

        if isinstance(raw, dict):
            score = raw.get("score", 0)
        else:
            score = 0

        score = _clamp_int(score, 0, 10)
        dimension_scores[dim] = score

        weighted_sum += score * weight
        total_weight += weight

    base_score = (weighted_sum / total_weight) * 10

    # Hard cap: if any critical dimension is tanked, cap the total
    for dim in CRITICAL_DIMENSIONS:
        if dimension_scores.get(dim, 10) < CRITICAL_FLOOR_THRESHOLD:
            base_score = min(base_score, CRITICAL_FLOOR_CAP)
            break

    return round(base_score, 2), dimension_scores


def apply_heuristic_adjustments(
    base_score: float,
    dimension_scores: dict,
    heuristics: dict
) -> tuple[float, list[str]]:
    """
    Heuristics act as soft multipliers on the base score.
    They can reduce but never increase the LLM score.
    Only fires an override when heuristic evidence contradicts
    LLM optimism — not as a flat penalty on all cases.
    """
    adjustment_notes = []
    multiplier = 1.0

    if not heuristics.get("has_endpoint", False):
        if dimension_scores.get("technical_correctness", 0) >= 7:
            multiplier *= 0.75
            adjustment_notes.append(
                "Heuristic override: LLM overscored technical_correctness,"
                " no endpoint found"
            )
        else:
            multiplier *= 0.90

    if not heuristics.get("has_assertions", False):
        if dimension_scores.get("assertions_testable", 0) >= 6:
            multiplier *= 0.72
            adjustment_notes.append(
                "Heuristic override: LLM overscored assertions_testable,"
                " no assertions found"
            )
        else:
            multiplier *= 0.92

    if not heuristics.get("has_payload", False):
        multiplier *= 0.95
        adjustment_notes.append("Minor penalty: no request payload detected")

    vague_count = heuristics.get("vague_match_count", 0)
    if vague_count > 0:
        vague_multiplier = max(0.70, 1.0 - (vague_count * 0.06))
        multiplier *= vague_multiplier
        adjustment_notes.append(
            f"Vague wording penalty: {vague_count} match(es)"
            f" → ×{vague_multiplier:.2f}"
        )

    if heuristics.get("has_edge_cases", False) and base_score >= 55:
        multiplier = min(1.08, multiplier * 1.08)
        adjustment_notes.append("Edge case bonus applied")

    if heuristics.get("exploit_detected", False):
        return 0.0, ["EXPLOIT DETECTED — score forced to 0"]

    adjusted = base_score * multiplier
    return round(max(0.0, min(100.0, adjusted)), 2), adjustment_notes


def filter_fabricated_critical_failures(
    critical_failures: list,
    heuristics: dict,
) -> tuple[list, list]:
    """
    Remove critical_failure strings that contradict heuristic evidence.
    Returns (filtered_failures, removed_failures).
    """
    removed = []
    filtered = []

    for failure in critical_failures:
        failure_lower = failure.lower()

        if (
            heuristics.get("has_endpoint", False)
            and "endpoint" in failure_lower
            and any(w in failure_lower for w in ["no ", "missing", "without", "no explicit"])
        ):
            removed.append(failure)
            continue

        if (
            heuristics.get("has_payload", False)
            and any(w in failure_lower for w in ["payload", "request body"])
            and any(w in failure_lower for w in ["no ", "missing", "without"])
        ):
            removed.append(failure)
            continue

        if (
            heuristics.get("has_assertions", False)
            and any(w in failure_lower for w in ["assertion", "expected response"])
            and any(w in failure_lower for w in ["no ", "missing", "without"])
        ):
            removed.append(failure)
            continue

        filtered.append(failure)

    return filtered, removed


def ensure_list(value):

    if isinstance(value, list):
        return value

    return []


def build_failure_response(reason: str, latency_ms=None):

    return {
        "evaluation_status": "judge_failed",
        "dimensions": {
            "technical_correctness": {"score": 0, "reason": reason},
            "automation_ready":      {"score": 0, "reason": reason},
            "assertions_testable":   {"score": 0, "reason": reason},
            "oracle_correct":        {"score": 0, "reason": reason},
            "edge_case_realism":     {"score": 0, "reason": reason},
        },
        "summary": f"Judge failed: {reason}",
        "strengths": [],
        "weaknesses": ["Judge evaluation could not be completed"],
        "critical_failures": [reason],
        "suggested_tags": ["judge-failure"],
        "confidence": 0,
        "judge_latency_ms": latency_ms,
        "judge_model": JUDGE_MODEL,
        "overall_score": 0,
        "quality_grade": "F",
        "adjustment_notes": [],
        "technical_correctness": "0",
        "automation_ready": "0",
        "assertions_testable": "0",
        "oracle_correct": "0",
        "edge_case_realism": "0",
    }


# ---------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------


def evaluate_output(scenario, generated_output):

    logger.info("Starting evaluation")

    scenario_name = scenario.get("name", "Unknown Scenario")
    feature = scenario.get("feature", "Unknown Feature")
    scenario_type = scenario.get("type", "Unknown Type")
    difficulty = scenario.get("difficulty", "Unknown")

    if not generated_output or not generated_output.strip():

        return {
            "evaluation_status": "invalid_input",
            "dimensions": {
                "technical_correctness": {"score": 0, "reason": "Empty generated output"},
                "automation_ready":      {"score": 0, "reason": "Empty generated output"},
                "assertions_testable":   {"score": 0, "reason": "Empty generated output"},
                "oracle_correct":        {"score": 0, "reason": "Empty generated output"},
                "edge_case_realism":     {"score": 0, "reason": "Empty generated output"},
            },
            "summary": "Generated output is empty.",
            "strengths": [],
            "weaknesses": ["No generated test cases provided"],
            "critical_failures": ["Empty generated output"],
            "suggested_tags": ["invalid-input"],
            "confidence": 0,
            "judge_latency_ms": 0,
            "judge_model": JUDGE_MODEL,
            "overall_score": 0,
            "quality_grade": "F",
            "adjustment_notes": [],
            "technical_correctness": "0",
            "automation_ready": "0",
            "assertions_testable": "0",
            "oracle_correct": "0",
            "edge_case_realism": "0",
        }

    generated_output = generated_output[:MAX_GENERATED_CHARS]

    # ---------------------------------------------------
    # HEURISTICS
    # ---------------------------------------------------

    heuristics = evaluate_heuristics(generated_output)

    prompt = f"""You are an extremely strict evaluator for AI-generated software test cases.

Your job is to detect weak, vague, manual-only, insecure, or non-automatable test cases.

SCENARIO

Name: {scenario_name}
Feature: {feature}
Type: {scenario_type}
Difficulty: {difficulty}

GENERATED TEST CASES

{generated_output}

STRICT RULES

A testcase WITHOUT:
- explicit endpoint
- request payload
- expected response
- assertions

must NEVER receive high scores.

Vague phrases like "works correctly", "verify functionality", "behaves properly", "looks good"
must be penalized heavily.

Security exploits must NEVER be considered valid passing behavior.

Automation-ready means executable, machine-verifiable, and deterministic.

{SCORING_ANCHORS}

{JUDGE_SCHEMA}"""

    last_error = None

    for attempt in range(1, JUDGE_MAX_RETRIES + 1):

        try:

            t0 = time.time()

            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={
                    "type": "json_object"
                },
                temperature=0,
                max_tokens=MAX_JUDGE_TOKENS,
                timeout=JUDGE_TIMEOUT_SECONDS,
            )

            latency_ms = round((time.time() - t0) * 1000)

            content = response.choices[0].message.content

            parsed, parse_error = safe_json_extract(content)

            if parse_error:
                return build_failure_response(parse_error, latency_ms)

            # ---------------------------------------------------
            # SAFETY NORMALIZATION
            # ---------------------------------------------------

            parsed["strengths"] = ensure_list(parsed.get("strengths"))
            parsed["weaknesses"] = ensure_list(parsed.get("weaknesses"))
            parsed["critical_failures"] = ensure_list(parsed.get("critical_failures"))
            parsed["suggested_tags"] = ensure_list(parsed.get("suggested_tags"))

            if not isinstance(parsed.get("dimensions"), dict):
                return build_failure_response("Missing dimensions", latency_ms)

            dims = parsed["dimensions"]

            parsed["dimensions"] = {
                "technical_correctness": _normalize_dimension(
                    dims.get("technical_correctness"), "technical_correctness"
                ),
                "automation_ready": _normalize_dimension(
                    dims.get("automation_ready"), "automation_ready"
                ),
                "assertions_testable": _normalize_dimension(
                    dims.get("assertions_testable"), "assertions_testable"
                ),
                "oracle_correct": _normalize_dimension(
                    dims.get("oracle_correct"), "oracle_correct"
                ),
                "edge_case_realism": _normalize_dimension(
                    dims.get("edge_case_realism"), "edge_case_realism"
                ),
            }

            parsed["confidence"] = _clamp_int(parsed.get("confidence"), 0, 10)

            # ---------------------------------------------------
            # CONSISTENCY CORRECTIONS
            # Fixes known LLM calibration failures before scoring.
            # ---------------------------------------------------

            parsed["dimensions"], consistency_corrections = apply_consistency_corrections(
                parsed["dimensions"],
                generated_output,
                heuristics,
            )

            parsed["critical_failures"], removed_failures = (
                filter_fabricated_critical_failures(
                    parsed["critical_failures"],
                    heuristics,
                )
            )

            if removed_failures:
                parsed.setdefault("adjustment_notes", []).append(
                    f"Removed {len(removed_failures)} fabricated critical failure(s): "
                    + "; ".join(removed_failures)
                )

            parsed["evaluation_status"] = "success"
            parsed["judge_latency_ms"] = latency_ms
            parsed["judge_model"] = JUDGE_MODEL

            # ---------------------------------------------------
            # SCORE + QUALITY
            # ---------------------------------------------------

            base_score, dimension_scores = normalize_score(parsed)

            adjusted_score, adjustment_notes = apply_heuristic_adjustments(
                base_score,
                dimension_scores,
                heuristics,
            )

            # Confidence-weighted dampening:
            # low LLM confidence pulls final score toward 50 (regression to mean)
            confidence_raw = parsed.get("confidence", 5)
            confidence = max(1, min(10, int(confidence_raw))) / 10.0
            confidence_dampening = 0.3

            final_score = (
                adjusted_score * confidence
                + 50.0 * (1 - confidence) * confidence_dampening
                + adjusted_score * (1 - confidence_dampening)
            ) / (confidence + (1 - confidence_dampening))

            final_score = round(max(0.0, min(100.0, final_score)))

            parsed["overall_score"] = int(final_score)
            parsed["quality_grade"] = get_quality_grade(parsed["overall_score"])
            parsed["dimension_scores"] = dimension_scores
            existing_notes = parsed.get("adjustment_notes", [])
            parsed["adjustment_notes"] = (
                existing_notes + consistency_corrections + adjustment_notes
            )

            # ---------------------------------------------------
            # ADD HEURISTIC ISSUES
            # ---------------------------------------------------

            parsed["weaknesses"].extend(heuristics["issues"])
            parsed["weaknesses"].extend(adjustment_notes)

            # ---------------------------------------------------
            # EXPLOIT OVERRIDE
            # ---------------------------------------------------

            if heuristics.get("exploit_detected", False):

                for key in parsed["dimensions"]:
                    parsed["dimensions"][key]["score"] = 0
                    parsed["dimensions"][key]["reason"] = (
                        "Security exploit treated as passing behavior"
                    )

                parsed["overall_score"] = 0
                parsed["quality_grade"] = "F"

                parsed["critical_failures"].append(
                    "Security exploit treated as passing behavior"
                )
                parsed["weaknesses"].append(
                    "Security exploit treated as passing behavior"
                )
                parsed["suggested_tags"].extend([
                    "security-exploit",
                    "manual-only",
                ])
                parsed["summary"] = (
                    "Security exploit treated as passing behavior"
                )

            # Flat string fields for backward-compat with Pydantic schema
            parsed["technical_correctness"] = str(
                parsed["dimensions"]["technical_correctness"]["score"]
            )
            parsed["automation_ready"] = str(
                parsed["dimensions"]["automation_ready"]["score"]
            )
            parsed["assertions_testable"] = str(
                parsed["dimensions"]["assertions_testable"]["score"]
            )
            parsed["oracle_correct"] = str(
                parsed["dimensions"]["oracle_correct"]["score"]
            )
            parsed["edge_case_realism"] = str(
                parsed["dimensions"]["edge_case_realism"]["score"]
            )

            logger.info("Evaluation completed in %sms", latency_ms)

            return parsed

        except Exception as e:

            last_error = e

            logger.error(
                "Judge attempt %s failed: %s",
                attempt,
                str(e),
            )

            if is_rate_limit_error(e) and attempt < JUDGE_MAX_RETRIES:
                time.sleep(JUDGE_BACKOFF_SECONDS * attempt)
                continue

            if is_timeout_error(e) and attempt < JUDGE_MAX_RETRIES:
                time.sleep(JUDGE_BACKOFF_SECONDS * attempt)
                continue

            break

    return build_failure_response(str(last_error))