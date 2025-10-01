"""
rag.py
------

This module defines the **RAG (Retrieval-Augmented Generation)** service
for the CoSMIC Coaching Writer. It retrieves relevant context from the
vector database to augment user queries before they are sent to the LLM.

Key responsibilities:
- Perform similarity search against the vector database.
- Filter documents by a relevance score threshold.
- Provide a structured context string, relevance scores, and source metadata.
"""

from .base import ServiceBase
from .vector_database import vector_db_singleton
from ..core.config import settings


class RAG(ServiceBase):
    """
    Retrieval-augmented generation service.

    Attributes:
        topk (int): Maximum number of documents to retrieve.
        threshold (float): Minimum relevance score to include a document.
        vector_db (VectorDatabase): Singleton vector database instance.
    """

    def __init__(self):
        super().__init__()
        self.topk = settings.retrieve_topk
        self.threshold = settings.retrieve_score_threshold
        self.vector_db = vector_db_singleton

    def __call__(self, query: str):
        """
        Perform a retrieval query.

        Args:
            query (str): User query string.

        Returns:
            Tuple[str, List[float], List[dict]]:
                - context (str): Concatenated snippets with IDs.
                - scores (List[float]): Relevance scores for each candidate.
                - kept (List[dict]): Structured source metadata with id, score, text, threshold flag.
        """
        docs = self.vector_db.similarity_search_with_relevance_scores(query, k=self.topk)
        kept = []
        scores = []

        for idx, (doc, score) in enumerate(docs):
            scores.append(score)
            if score >= self.threshold:
                snippet = doc.page_content.replace('\n', ' ')
                kept.append({
                    "id": idx,
                    "score": score,
                    "text": snippet[:600],
                    "passed_threshold": True,
                })

        # Fallback: always return the top document if nothing passed threshold
        if not kept and docs:
            top_doc, top_score = docs[0]
            kept.append({
                "id": 0,
                "score": top_score,
                "text": top_doc.page_content.replace('\n', ' ')[:600],
                "passed_threshold": False,
            })

        context = "\n".join([f"[{k['id']}] {k['text']}" for k in kept])
        return context, scores, kept


# Singleton instance
rag_singleton = RAG()