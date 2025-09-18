from .base import ServiceBase
from .llm import llm_singleton
from .rag import rag_singleton


class CoachingService(ServiceBase):
    def __init__(self):
        super().__init__()
        self.llm = llm_singleton
        self.rag = rag_singleton

    def build_prompt(self, query: str, mode: str | None = None):
        prefix = "" if not mode else f"[Mode: {mode}] "
        return prefix + query

    def __call__(self, query: str, use_rag: bool = True, mode: str | None = None):
        context = ""
        retrieve_scores = []
        sources = []
        if use_rag:
            context, retrieve_scores, sources = self.rag(query)
        prompt = self.build_prompt(query, mode=mode)
        response, raw = self.llm(prompt, context=context)
        if sources and use_rag:
            marker_list = ", ".join([f"[{s['id']}]" for s in sources])
            if marker_list not in response:
                response += f"\n\nSources: {marker_list}"
        return {
            "response": response,
            "raw_response": raw,
            "retrieve_scores": retrieve_scores,
            "used_context": context,
            "sources": sources,
        }


coach_singleton = CoachingService()
