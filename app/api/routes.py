from fastapi import APIRouter
from app.schemas.response import EvaluationResponse
from app.schemas.request import EvaluationRequest
from app.services.evaluator import evaluate_output
from app.services.scorer import calculate_overall_score

router = APIRouter()

@router.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": "ai-testcase-validator",
        "judge_model": "llama-3.1-8b-instant"
    }

@router.post(
    "/evaluate",
    response_model=EvaluationResponse
)
async def evaluate(req: EvaluationRequest):

    scenario = {
        "name": req.scenario_name,
        "feature": req.feature,
        "type": req.scenario_type,
        "difficulty": req.difficulty
    }

    evaluation = evaluate_output(
        scenario,
        req.generated_output
    )
    evaluation["overall_score"] = calculate_overall_score(evaluation)
    return evaluation