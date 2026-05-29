from pydantic import BaseModel
from typing import List


class EvaluationResponse(BaseModel):

    evaluation_status: str

    technical_correctness: str
    automation_ready: str
    assertions_testable: str
    oracle_correct: str
    edge_case_realism: str

    summary: str

    strengths: List[str]
    weaknesses: List[str]
    critical_failures: List[str]

    suggested_tags: List[str]

    judge_latency_ms: int | None = None
    judge_model: str | None = None

    overall_score: int | None = None