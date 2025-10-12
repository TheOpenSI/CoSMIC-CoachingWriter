"""
requests.py
------------

Defines the **Pydantic request models** for all API routes in the CoSMIC
Coaching Writer service. These ensure consistent structure and validation
for incoming client requests.

### Supported Endpoints
- `/coach/query`
- `/documents/ingest-text`
"""

from pydantic import BaseModel
from typing import Any, List, Union, Optional


class QueryRequest(BaseModel):
    """
    Request body for `/coach/query`.

    Attributes:
        query (str): User’s query or academic draft text.
        use_rag (bool): Whether to apply Retrieval-Augmented Generation.
        mode (Optional[str]): Optional task mode (e.g., "critique", "references").
        documents (Optional[List[Union[str, dict]]]): Optional document metadata
            coming from OpenWebUI attachments. Can include absolute paths or
            dictionaries containing a `path` or `name` key.
    """
    query: str
    use_rag: bool = True
    mode: Optional[str] = None
    documents: Optional[List[Union[str, dict]]] = None


class TextIngestRequest(BaseModel):
    """
    Request body for `/documents/ingest-text`.

    Attributes:
        text (str): Raw text to ingest into the vector database for retrieval.
    """
    text: str
