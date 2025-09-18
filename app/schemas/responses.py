from pydantic import BaseModel, Field
from typing import List


class HealthResponse(BaseModel):
    status: str
    llm: str


class QueryResponse(BaseModel):
    response: str
    raw_response: str
    retrieve_scores: List[float] = Field(default_factory=list)
    used_context: str = ""
