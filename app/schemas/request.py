from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):

    scenario_name: str = Field(min_length=3)

    feature: str = Field(min_length=3)

    scenario_type: str = Field(min_length=3)

    difficulty: str = Field(min_length=3)

    generated_output: str = Field(min_length=10)