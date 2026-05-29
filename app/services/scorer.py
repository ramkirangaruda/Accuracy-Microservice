import re


WEIGHTS = {
    "coverage": 0.20,
    "relevance": 0.25,
    "structure": 0.20,
    "edge_cases": 0.15,
    "clarity": 0.10,
    "consistency": 0.10,
}


def safe_str(value):

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (dict, list)):
        return str(value)

    return str(value)


def safe_join_steps(steps):

    if steps is None:
        return ""

    if isinstance(steps, list):
        return " ".join(
            safe_str(s)
            for s in steps
        )

    return safe_str(steps)


def calculate_overall_score(evaluation: dict) -> int:

    score = 0

    technical_map = {
        "yes": 30,
        "partial": 15,
        "no": 0
    }

    automation_map = {
        "yes": 25,
        "partial": 12,
        "no": 0
    }

    assertions_map = {
        "yes": 20,
        "partial": 10,
        "no": 0
    }

    oracle_map = {
        "yes": 15,
        "partial": 7,
        "no": 0
    }

    edge_map = {
        "excellent": 10,
        "good": 7,
        "partial": 4,
        "poor": 0
    }

    score += technical_map.get(
        evaluation.get("technical_correctness", "no"),
        0
    )

    score += automation_map.get(
        evaluation.get("automation_ready", "no"),
        0
    )

    score += assertions_map.get(
        evaluation.get("assertions_testable", "no"),
        0
    )

    score += oracle_map.get(
        evaluation.get("oracle_correct", "no"),
        0
    )

    score += edge_map.get(
        evaluation.get("edge_case_realism", "poor"),
        0
    )

    return score


def score(parsed: dict, feature: str, requested_count: int) -> dict:

    # Handle malformed top-level structures safely
    if isinstance(parsed, dict):
        tc_list = parsed.get("test_cases", [])
    elif isinstance(parsed, list):
        # Some models return raw list instead of object
        tc_list = parsed
    else:
        tc_list = []

    if not isinstance(tc_list, list):
        tc_list = []

    # Keep only dict objects
    tc_list = [
        tc for tc in tc_list
        if isinstance(tc, dict)
    ]

    n = len(tc_list)

    if n == 0:
        return {
            "coverage": 0,
            "relevance": 0,
            "structure": 0,
            "edge_cases": 0,
            "clarity": 0,
            "consistency": 0,
            "overall": 0,
            "count": 0,
        }

    # ─────────────────────────────────────────────
    # Duplicate Penalty
    # ─────────────────────────────────────────────

    titles = [
        safe_str(
            tc.get("title", "")
        ).strip().lower()
        for tc in tc_list
    ]

    unique_titles = len(set(titles))

    dup_penalty = max(
        0,
        (n - unique_titles) * 10
    )

    # ─────────────────────────────────────────────
    # Coverage
    # ─────────────────────────────────────────────

    if requested_count <= 0:
        coverage = 0
    else:
        coverage = min(
            100,
            round((n / requested_count) * 100)
        )

    # ─────────────────────────────────────────────
    # Relevance
    # ─────────────────────────────────────────────

    feature_words = [
        w for w in re.split(r"\W+", feature.lower())
        if len(w) > 3
    ]

    rel_scores = []

    for tc in tc_list:

        blob = " ".join([

            safe_str(
                tc.get("title", "")
            ),

            safe_join_steps(
                tc.get("steps", [])
            ),

            safe_str(
                tc.get("expected_result", "")
            ),

            safe_str(
                tc.get("preconditions", "")
            ),

        ]).lower()

        hits = sum(
            1 for w in feature_words
            if w in blob
        )

        threshold = max(
            1,
            len(feature_words) * 0.25
        )

        rel_scores.append(
            min(1.0, hits / threshold)
        )

    relevance = round(
        (sum(rel_scores) / n) * 100
    )

    # ─────────────────────────────────────────────
    # Structure
    # ─────────────────────────────────────────────

    struct_scores = []

    for tc in tc_list:

        s = 0

        if tc.get("id"):
            s += 5

        title = safe_str(
            tc.get("title", "")
        )

        if len(title) >= 8:
            s += 15

        if tc.get("type") in (
            "positive",
            "negative",
            "edge"
        ):
            s += 10

        if tc.get("priority") in (
            "high",
            "medium",
            "low"
        ):
            s += 10

        preconditions = safe_str(
            tc.get("preconditions", "")
        )

        if len(preconditions) > 5:
            s += 15

        steps = tc.get("steps", [])

        if isinstance(steps, list):

            valid_steps = [
                s for s in steps
                if isinstance(s, str)
            ]

            if len(valid_steps) >= 2:
                s += 25

        expected_result = safe_str(
            tc.get("expected_result", "")
        )

        if len(expected_result) > 10:
            s += 15

        if tc.get("category"):
            s += 5

        struct_scores.append(s)

    structure = round(
        sum(struct_scores) / n
    )

    # ─────────────────────────────────────────────
    # Edge Cases
    # ─────────────────────────────────────────────

    edge_pattern = re.compile(
        r"edge|boundary|invalid|null|empty|max|min|overflow|exceed|zero|none|missing|corrupt",
        re.IGNORECASE
    )

    neg_pattern = re.compile(
        r"fail|error|wrong|unauthori|deny|reject|block|timeout|expired|missing",
        re.IGNORECASE
    )

    edge_count = sum(
        1 for tc in tc_list
        if tc.get("type") == "edge"
        or edge_pattern.search(
            safe_str(
                tc.get("title", "")
            )
        )
    )

    neg_count = sum(
        1 for tc in tc_list
        if tc.get("type") == "negative"
        or neg_pattern.search(
            safe_str(
                tc.get("title", "")
            )
        )
    )

    raw_ratio = (
        edge_count + neg_count * 0.7
    ) / n

    edge_cases = min(
        100,
        round(raw_ratio * 180)
    )

    # ─────────────────────────────────────────────
    # Diversity
    # ─────────────────────────────────────────────

    types = [
        tc.get("type")
        for tc in tc_list
    ]

    type_diversity = len(set(types))

    diversity_penalty = max(
        0,
        (3 - type_diversity) * 15
    )

    # ─────────────────────────────────────────────
    # Clarity
    # ─────────────────────────────────────────────

    generic_phrases = [
        "verify",
        "check",
        "ensure",
        "validate"
    ]

    clarity_scores = []

    for tc in tc_list:

        c = 50

        title = safe_str(
            tc.get("title", "")
        )

        title_len = len(title)

        if 10 <= title_len <= 90:
            c += 20

        steps = tc.get("steps", [])

        step_blob = safe_join_steps(
            steps
        ).lower()

        generic_hits = sum(
            1 for g in generic_phrases
            if g in step_blob
        )

        c -= generic_hits * 3

        if (
            isinstance(steps, list)
            and all(
                isinstance(s, str)
                and len(s) > 8
                for s in steps
            )
        ):
            c += 20

        er = safe_str(
            tc.get("expected_result", "")
        )

        if len(er) > 15:
            c += 10

        clarity_scores.append(c)

    clarity = round(
        sum(clarity_scores) / n
    )

    # ─────────────────────────────────────────────
    # Consistency
    # ─────────────────────────────────────────────

    consistency = 100

    # ─────────────────────────────────────────────
    # Overall
    # ─────────────────────────────────────────────

    overall = round(

        coverage * WEIGHTS["coverage"] +
        relevance * WEIGHTS["relevance"] +
        structure * WEIGHTS["structure"] +
        edge_cases * WEIGHTS["edge_cases"] +
        clarity * WEIGHTS["clarity"] +
        consistency * WEIGHTS["consistency"]

        - dup_penalty
        - diversity_penalty
    )

    overall = max(
        0,
        min(100, overall)
    )

    return {

        "coverage": coverage,

        "relevance": relevance,

        "structure": structure,

        "edge_cases": edge_cases,

        "clarity": clarity,

        "consistency": consistency,

        "overall": overall,

        "count": n,

        "positive_count": sum(
            1 for tc in tc_list
            if tc.get("type") == "positive"
        ),

        "negative_count": sum(
            1 for tc in tc_list
            if tc.get("type") == "negative"
        ),

        "edge_count": sum(
            1 for tc in tc_list
            if tc.get("type") == "edge"
        ),
    }