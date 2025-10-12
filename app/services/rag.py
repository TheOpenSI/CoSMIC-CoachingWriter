"""
rag.py
------

Defines the **Retrieval-Augmented Generation (RAG)** subsystem for the
CoSMIC Coaching Writer.

This service retrieves semantically relevant content from the vector database
to provide factual or conceptual grounding for the LLM before inference.

### Responsibilities
- Query the FAISS vector store for semantically similar documents.
- Filter out low-relevance results below a configurable threshold.
- Format retrieved passages into a structured “context” string
  that the LLM can interpret effectively.

### Environment integration
Uses retrieval parameters defined in the global `settings` object:
- `retrieve_topk`: Number of candidates to consider.
- `retrieve_score_threshold`: Minimum cosine similarity to retain a document.

### Returns
A triple of `(context, scores, kept)` where:
- `context` is a newline-joined text corpus (with doc IDs)
- `scores` is a list of numeric relevance values
- `kept` is structured metadata for each selected document
"""

from .base import ServiceBase
from .vector_database import vector_db_singleton
from ..core.config import settings


class RAG(ServiceBase):
    """Retrieval-augmented generation helper for context fetching."""

    def __init__(self):
        super().__init__()
        self.topk = settings.retrieve_topk
        self.threshold = settings.retrieve_score_threshold
        self.vector_db = vector_db_singleton

    def __call__(self, query: str):
        """
        Retrieve context snippets for a given query.

        Args:
            query (str): User’s message or draft text.

        Returns:
            tuple[str, list[float], list[dict]]: 
              (context, scores, kept) — where `kept` contains:
                {
                    "id": int,
                    "score": float,
                    "text": str,
                    "passed_threshold": bool
                }
        """
        docs = self.vector_db.similarity_search_with_relevance_scores(query, k=self.topk)
        kept, scores = [], []

        for idx, (doc, score) in enumerate(docs):
            scores.append(score)
            if score >= self.threshold:
                kept.append({
                    "id": idx,
                    "score": score,
                    "text": doc.page_content.replace('\n', ' ')[:600],
                    "passed_threshold": True,
                })

        # Fallback to top document if no match meets threshold
        if not kept and docs:
            top_doc, top_score = docs[0]
            kept.append({
                "id": 0,
                "score": top_score,
                "text": top_doc.page_content.replace('\n', ' ')[:600],
                "passed_threshold": False,
            })

        context = "\n".join(f"[{k['id']}] {k['text']}" for k in kept)
        return context, scores, kept


# Singleton instance for reuse across app
rag_singleton = RAG()
