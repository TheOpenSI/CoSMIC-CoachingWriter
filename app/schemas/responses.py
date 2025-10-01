"""
Pydantic Response Schemas

Defines structured response models used in API responses.
"""

from pydantic import BaseModel
from typing import List, Optional


class HealthResponse(BaseModel):
    """
    Response body for /health endpoint.

    Attributes:
        status (str): Health status ("ok" if healthy).
        llm (str): Name of configured LLM.
    """
    status: str
    llm: str


class QueryResponse(BaseModel):
    """
    Response body for /coach/query endpoint.

    Attributes:
        response (str): Final processed coaching response.
        raw_response (Optional[str]): Raw unprocessed LLM response.
        retrieve_scores (List[float]): Similarity scores of retrieved docs.
        used_context (str): Concatenated context snippets fed into LLM.
    """
    response: str
    raw_response: Optional[str] = None
    retrieve_scores: List[float] = []
    used_context: str = ""