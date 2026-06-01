import re


def evaluate_heuristics(text: str) -> dict:

    text_lower = text.lower()

    penalties = 0
    issues = []

    # ---------------------------------------------------
    # ENDPOINT DETECTION
    # ---------------------------------------------------

    endpoint_patterns = [
        r"post\s+/",
        r"get\s+/",
        r"put\s+/",
        r"delete\s+/",
        r"patch\s+/",
        r"/api/",
    ]

    has_endpoint = any(
        re.search(pattern, text_lower)
        for pattern in endpoint_patterns
    )

    if not has_endpoint:
        penalties += 8
        issues.append("Missing explicit API endpoint")

    # ---------------------------------------------------
    # ASSERTION DETECTION
    # ---------------------------------------------------

    assertion_keywords = [
        "assert",
        "expected",
        "status",
        "response contains",
        "response time",
        "status code",
        "verify response",
    ]

    has_assertions = any(
        keyword in text_lower
        for keyword in assertion_keywords
    )

    if not has_assertions:
        penalties += 10
        issues.append("Missing machine-verifiable assertions")

    # ---------------------------------------------------
    # PAYLOAD DETECTION
    # ---------------------------------------------------

    payload_patterns = [
        "{",
        "payload",
        "request body",
        '"email"',
        '"password"',
    ]

    has_payload = any(
        pattern in text_lower
        for pattern in payload_patterns
    )

    if not has_payload:
        penalties += 6
        issues.append("Missing request payload details")

    # ---------------------------------------------------
    # VAGUE WORDING
    # ---------------------------------------------------

    vague_phrases = [
        "works correctly",
        "looks good",
        "user friendly",
        "verify functionality",
        "behaves properly",
        "works properly",
        "functions correctly",
        "displayed correctly",
    ]

    vague_matches = [
        phrase
        for phrase in vague_phrases
        if phrase in text_lower
    ]

    vague_match_count = len(vague_matches)

    if vague_matches:
        penalties += 5 * vague_match_count
        issues.append("Contains vague non-testable wording")

    # ---------------------------------------------------
    # EDGE CASE DETECTION
    # ---------------------------------------------------

    edge_case_keywords = [
        "invalid",
        "expired",
        "empty",
        "timeout",
        "rate limit",
        "401",
        "403",
        "404",
        "500",
    ]

    has_edge_cases = any(
        keyword in text_lower
        for keyword in edge_case_keywords
    )

    # Reward realistic edge-case thinking
    if has_edge_cases:
        penalties -= 5

    # ---------------------------------------------------
    # SECURITY EXPLOIT DETECTION
    # ---------------------------------------------------

    exploit_patterns = [
        "' or '1'='1",
        "sql injection",
        "bypass login",
        "unauthorized access is granted",
        "successful exploit",
    ]

    exploit_detected = any(
        pattern in text_lower
        for pattern in exploit_patterns
    )

    if exploit_detected:
        penalties += 100
        issues.append("Security exploit treated as valid behavior")

    # ---------------------------------------------------
    # FINAL CLAMP
    # ---------------------------------------------------

    penalties = max(0, penalties)

    return {
        "penalties": penalties,
        "issues": issues,
        "has_endpoint": has_endpoint,
        "has_assertions": has_assertions,
        "has_payload": has_payload,
        "has_edge_cases": has_edge_cases,
        "exploit_detected": exploit_detected,
        # Exposed so apply_heuristic_adjustments can scale
        # the vague-wording penalty continuously instead of binary
        "vague_match_count": vague_match_count,
    }