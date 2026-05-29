import os
import json
import time
import re

from dotenv import load_dotenv
from openai import OpenAI

from app.services.heuristics import evaluate_heuristics
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
MAX_JUDGE_TOKENS = 500
JUDGE_TIMEOUT_SECONDS = 20

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


def normalize_score(parsed: dict) -> int:

    score = 100

    if parsed.get("technical_correctness") == "partial":
        score -= 20

    if parsed.get("technical_correctness") == "no":
        score -= 50

    if parsed.get("automation_ready") == "partial":
        score -= 15

    if parsed.get("automation_ready") == "no":
        score -= 35

    if parsed.get("assertions_testable") == "partial":
        score -= 15

    if parsed.get("assertions_testable") == "no":
        score -= 30

    if parsed.get("oracle_correct") == "partial":
        score -= 20

    if parsed.get("oracle_correct") == "no":
        score -= 40

    realism = parsed.get("edge_case_realism")

    if realism == "good":
        score -= 5

    elif realism == "partial":
        score -= 15

    elif realism == "poor":
        score -= 30

    return max(0, min(100, score))


def ensure_list(value):

    if isinstance(value, list):
        return value

    return []


def build_failure_response(reason: str, latency_ms=None):

    return {
        "evaluation_status": "judge_failed",
        "technical_correctness": "unknown",
        "automation_ready": "unknown",
        "assertions_testable": "unknown",
        "oracle_correct": "unknown",
        "edge_case_realism": "unknown",
        "summary": f"Judge failed: {reason}",
        "strengths": [],
        "weaknesses": [
            "Judge evaluation could not be completed"
        ],
        "critical_failures": [
            reason
        ],
        "suggested_tags": [
            "judge-failure"
        ],
        "judge_latency_ms": latency_ms,
        "judge_model": JUDGE_MODEL,
        "overall_score": 0
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
            "technical_correctness": "no",
            "automation_ready": "no",
            "assertions_testable": "no",
            "oracle_correct": "no",
            "edge_case_realism": "poor",
            "summary": "Generated output is empty.",
            "strengths": [],
            "weaknesses": [
                "No generated test cases provided"
            ],
            "critical_failures": [
                "Empty generated output"
            ],
            "suggested_tags": [
                "invalid-input"
            ],
            "judge_latency_ms": 0,
            "judge_model": JUDGE_MODEL,
            "overall_score": 0
        }

    generated_output = generated_output[:MAX_GENERATED_CHARS]

    # ---------------------------------------------------
    # HEURISTICS
    # ---------------------------------------------------

    heuristics = evaluate_heuristics(generated_output)

    prompt = f"""
You are an extremely strict evaluator for AI-generated software test cases.

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

Vague phrases like:
- works correctly
- verify functionality
- behaves properly
- looks good

must be penalized heavily.

Security exploits must NEVER be considered valid passing behavior.

Automation-ready means:
- executable
- machine-verifiable
- deterministic

Return ONLY valid JSON.

JSON SCHEMA:

{{
    "technical_correctness": "",
    "automation_ready": "",
    "assertions_testable": "",
    "oracle_correct": "",
    "edge_case_realism": "",
    "summary": "",
    "strengths": [],
    "weaknesses": [],
    "critical_failures": [],
    "suggested_tags": []
}}
"""

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

                return build_failure_response(
                    parse_error,
                    latency_ms
                )

            # ---------------------------------------------------
            # SAFETY NORMALIZATION
            # ---------------------------------------------------

            parsed["strengths"] = ensure_list(
                parsed.get("strengths")
            )

            parsed["weaknesses"] = ensure_list(
                parsed.get("weaknesses")
            )

            parsed["critical_failures"] = ensure_list(
                parsed.get("critical_failures")
            )

            parsed["suggested_tags"] = ensure_list(
                parsed.get("suggested_tags")
            )

            parsed["evaluation_status"] = "success"

            parsed["judge_latency_ms"] = latency_ms
            parsed["judge_model"] = JUDGE_MODEL

            # ---------------------------------------------------
            # BASE SCORE
            # ---------------------------------------------------

            base_score = normalize_score(parsed)

            # ---------------------------------------------------
            # APPLY HEURISTIC PENALTIES
            # ---------------------------------------------------

            final_score = max(
                0,
                base_score - heuristics["penalties"]
            )

            parsed["overall_score"] = final_score

            # ---------------------------------------------------
            # ADD HEURISTIC ISSUES
            # ---------------------------------------------------

            parsed["weaknesses"].extend(
                heuristics["issues"]
            )

            # ---------------------------------------------------
            # EXPLOIT OVERRIDE
            # ---------------------------------------------------

            if heuristics["exploit_detected"]:

                parsed["technical_correctness"] = "no"

                parsed["automation_ready"] = "no"

                parsed["assertions_testable"] = "no"

                parsed["oracle_correct"] = "no"

                parsed["edge_case_realism"] = "poor"

                parsed["overall_score"] = 0

                parsed["critical_failures"].append(
                    "Security exploit treated as passing behavior"
                )

                parsed["weaknesses"].append(
                    "Security exploit treated as passing behavior"
                )

                parsed["suggested_tags"].extend([
                    "security-exploit",
                    "manual-only"
                ])

                parsed["summary"] = (
                    "Security exploit treated as passing behavior"
                )

            logger.info(
                "Evaluation completed in %sms",
                latency_ms
            )

            return parsed

        except Exception as e:

            last_error = e

            logger.error(
                "Judge attempt %s failed: %s",
                attempt,
                str(e)
            )

            if (
                is_rate_limit_error(e)
                and attempt < JUDGE_MAX_RETRIES
            ):

                time.sleep(
                    JUDGE_BACKOFF_SECONDS * attempt
                )

                continue

            if (
                is_timeout_error(e)
                and attempt < JUDGE_MAX_RETRIES
            ):

                time.sleep(
                    JUDGE_BACKOFF_SECONDS * attempt
                )

                continue

            break

    return build_failure_response(str(last_error))