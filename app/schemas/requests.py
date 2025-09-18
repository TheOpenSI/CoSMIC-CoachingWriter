from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or draft content.")
    use_rag: bool = Field(default=True, description="Enable retrieval-augmented generation.")
    mode: Optional[str] = Field(default=None, description="Coaching mode: critique, improve, structure, abstract, references")


class TextIngestRequest(BaseModel):
    text: str
