import re


def evaluate_heuristics(text: str) -> dict:
    text_lower = text.lower()

    penalties = 0
    issues = []

    # -------------------------
    # Missing endpoint
    # -------------------------
    endpoint_patterns = [
        r"post\s+/",
        r"get\s+/",
        r"put\s+/",
        r"delete\s+/",
        r"/api/"
    ]

    has_endpoint = any(
        re.search(pattern, text_lower)
        for pattern in endpoint_patterns
    )

    if not has_endpoint:
        penalties += 20
        issues.append("Missing explicit API endpoint")

    # -------------------------
    # Missing assertions
    # -------------------------
    assertion_keywords = [
        "assert",
        "expected",
        "verify",
        "status",
        "response"
    ]

    has_assertions = any(
        keyword in text_lower
        for keyword in assertion_keywords
    )

    if not has_assertions:
        penalties += 25
        issues.append("Missing machine-verifiable assertions")

    # -------------------------
    # Vague wording
    # -------------------------
    vague_phrases = [
        "works correctly",
        "looks good",
        "user friendly",
        "verify functionality",
        "behaves properly"
    ]

    found_vague = any(
        phrase in text_lower
        for phrase in vague_phrases
    )

    if found_vague:
        penalties += 15
        issues.append("Contains vague non-testable wording")

    # -------------------------
    # Security exploit normalization
    # -------------------------
    exploit_patterns = [
        "' or '1'='1",
        "sql injection",
        "bypass login",
        "unauthorized access"
    ]

    exploit_detected = any(
        pattern in text_lower
        for pattern in exploit_patterns
    )

    return {
        "penalties": penalties,
        "issues": issues,
        "has_endpoint": has_endpoint,
        "has_assertions": has_assertions,
        "exploit_detected": exploit_detected
    }