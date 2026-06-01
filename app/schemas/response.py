from pydantic import BaseModel
from typing import Dict, List, Optional


class EvaluationResponse(BaseModel):

    evaluation_status: str

    dimensions: Dict[str, "DimensionScore"]

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
    adjustment_notes: List[str] = []

    judge_latency_ms: Optional[int] = None
    judge_model: Optional[str] = None
    confidence: Optional[int] = None

    overall_score: int
    quality_grade: Optional[str] = None


class DimensionScore(BaseModel):

    score: int
    reason: str