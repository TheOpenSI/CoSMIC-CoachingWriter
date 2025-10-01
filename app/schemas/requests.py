"""
Pydantic Request Schemas

Defines the data models used as request bodies for API endpoints.
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """
    Request body for /coach/query endpoint.

    Attributes:
        query (str): User query or academic text to analyze.
        use_rag (bool): Whether to use retrieval-augmented generation.
        mode (str | None): Optional mode (e.g., "critique", "references").
    """
    query: str
    use_rag: bool = True
    mode: str | None = None


class TextIngestRequest(BaseModel):
    """
    Request body for /documents/ingest-text endpoint.

    Attributes:
        text (str): Raw text to ingest into the vector database.
    """
    text: str