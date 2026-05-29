import os
import json
import time
import re

from dotenv import load_dotenv
from openai import OpenAI

from app.core.logger import logger

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# -----------------------------
# CONFIG
# -----------------------------

JUDGE_MODEL = "llama-3.1-8b-instant"

JUDGE_MAX_RETRIES = 3
JUDGE_BACKOFF_SECONDS = 2

MAX_GENERATED_CHARS = 3000
MAX_JUDGE_TOKENS = 400
JUDGE_TIMEOUT_SECONDS = 20

# -----------------------------
# HELPERS
# -----------------------------


def safe_json_extract(text):

    if not text:
        return None, "Empty judge response"

    cleaned = text.strip()

    # remove markdown code fences
    if cleaned.startswith("```"):
        cleaned = (
            cleaned
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    candidates = [cleaned]

    # attempt object extraction
    object_match = re.search(r"\{[\s\S]+\}", cleaned)

    # attempt array extraction
    array_match = re.search(r"\[[\s\S]+\]", cleaned)

    if object_match:
        candidates.append(object_match.group())

    if array_match:
        candidates.append(array_match.group())

    for candidate in candidates:

        # remove trailing commas
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

        try:

            parsed = json.loads(candidate)

            if isinstance(parsed, list):
                return {
                    "evaluation_items": parsed
                }, None

            if isinstance(parsed, dict):
                return parsed, None

        except json.JSONDecodeError:
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
            "request timed out",
        ]
    )


def build_failure_response(reason: str, latency_ms=None):

    return {
        "evaluation_status": "judge_failed",
        "judge_model": JUDGE_MODEL,
        "judge_latency_ms": latency_ms,

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
        ]
    }


# -----------------------------
# MAIN EVALUATION FUNCTION
# -----------------------------


def evaluate_output(scenario, generated_output):

    logger.info("Starting evaluation")

    # defensive scenario handling
    scenario_name = scenario.get("name", "Unknown Scenario")
    feature = scenario.get("feature", "Unknown Feature")
    scenario_type = scenario.get("type", "Unknown Type")
    difficulty = scenario.get("difficulty", "Unknown")

    # validate generated output
    if not generated_output or not generated_output.strip():

        logger.warning("Generated output is empty")

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

            "judge_model": JUDGE_MODEL,
            "judge_latency_ms": 0
        }

    # prevent giant prompts
    generated_output = generated_output[:MAX_GENERATED_CHARS]

    prompt = f"""
You are evaluating AI-generated software test cases.

Evaluate STRICTLY according to the rubric below.

Scenario
Name: {scenario_name}
Feature: {feature}
Type: {scenario_type}
Difficulty: {difficulty}

Generated Test Cases
{generated_output}

Evaluation Rubric

technical_correctness
- yes = technically accurate and semantically correct
- partial = mostly correct but flawed assumptions
- no = fundamentally incorrect

automation_ready
- yes = directly scriptable with endpoints/payloads/assertions
- partial = partially scriptable
- no = manual QA only

assertions_testable
- yes = machine-verifiable assertions
- partial = vague assertions
- no = not objectively testable

oracle_correct
- yes = pass/fail logic correct
- partial = weak/inconsistent
- no = inverted/broken logic

edge_case_realism
- excellent = production-grade
- good = meaningful
- partial = generic
- poor = unrealistic

Important Rules
- Penalize hallucinated endpoints/features
- Penalize vague assertions
- Penalize manual-only workflows
- Penalize oracle inversion heavily
- Reward executable payloads/endpoints/assertions
- Reward realistic production edge cases
- Security exploits should NEVER be treated as passing behavior

Return ONLY valid JSON.

Schema:
{{
    "evaluation_status": "success",
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

    logger.info("[JUDGE] Using model: %s", JUDGE_MODEL)

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

                logger.error("JSON parse failed: %s", parse_error)

                return build_failure_response(
                    parse_error,
                    latency_ms=latency_ms
                )

            parsed["evaluation_status"] = "success"
            parsed["judge_latency_ms"] = latency_ms
            parsed["judge_model"] = JUDGE_MODEL

            logger.info(
                "Evaluation completed in %sms",
                latency_ms
            )

            return parsed

        except Exception as e:

            last_error = e

            logger.error(
                "[JUDGE] Attempt %s failed: %s",
                attempt,
                str(e)
            )

            if (
                is_rate_limit_error(e)
                and attempt < JUDGE_MAX_RETRIES
            ):

                logger.warning(
                    "[JUDGE] Rate limit hit. Retrying in %ss",
                    JUDGE_BACKOFF_SECONDS * attempt
                )

                time.sleep(JUDGE_BACKOFF_SECONDS * attempt)

                continue

            if (
                is_timeout_error(e)
                and attempt < JUDGE_MAX_RETRIES
            ):

                logger.warning(
                    "[JUDGE] Timeout. Retrying in %ss",
                    JUDGE_BACKOFF_SECONDS * attempt
                )

                time.sleep(JUDGE_BACKOFF_SECONDS * attempt)

                continue

            break

    logger.error("Judge completely failed")

    return build_failure_response(str(last_error))