"""
responses.py
-------------

Defines the **Pydantic response models** used across API endpoints in
the CoSMIC Coaching Writer service. These ensure standardized and
machine-readable responses for health checks and query results.
"""

from pydantic import BaseModel
from typing import List, Optional


class HealthResponse(BaseModel):
    """
    Response for `/health`.

    Attributes:
        status (str): Service status ("ok" when healthy).
        llm (str): Active LLM identifier configured in settings.
    """
    status: str
    llm: str


class QueryResponse(BaseModel):
    """
    Response for `/coach/query`.

    Attributes:
        response (str): Final formatted coaching feedback.
        raw_response (Optional[str]): Raw LLM output prior to filtering.
        retrieve_scores (List[float]): Vector similarity scores for retrieved documents.
        used_context (str): Text context provided to the LLM during inference.
    """
    response: str
    raw_response: Optional[str] = None
    retrieve_scores: List[float] = []
    used_context: str = ""
