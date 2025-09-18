from .base import ServiceBase
from .vector_database import vector_db_singleton
from ..core.config import settings


class RAG(ServiceBase):
    def __init__(self):
        super().__init__()
        self.topk = settings.retrieve_topk
        self.threshold = settings.retrieve_score_threshold
        self.vector_db = vector_db_singleton

    def __call__(self, query: str):
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


rag_singleton = RAG()
