"""
coach.py
--------

Implements the **CoachingService**, the orchestration layer that connects
retrieval and generation for the CoSMIC Coaching Writer.

### Overview
The CoachingService serves as the brain of the system:
  1. Receives a user’s query (draft or question)
  2. Optionally retrieves relevant supporting text using the RAG subsystem
  3. Builds a mode-specific prompt (chat or task)
  4. Calls the LLM to generate structured, rule-compliant feedback
  5. Returns the feedback with retrieval metadata

### Responsibilities
- Route and merge RAG context with user queries.
- Adjust tone and analytical style depending on `mode` ("chat" vs "task").
- Post-process model output to include source markers if context was used.
"""

from .base import ServiceBase
from .llm import llm_singleton
from .rag import rag_singleton


class CoachingService(ServiceBase):
    """
    Orchestrates retrieval and LLM calls for coaching responses.

    Attributes:
        llm (OllamaChat): LLM service wrapper.
        rag (RAG): Retrieval service wrapper.
    """

    def __init__(self):
        super().__init__()
        self.llm = llm_singleton
        self.rag = rag_singleton

    def build_prompt(self, query: str, mode: str | None = None):
        """
        Build a mode-specific prefix for the query.

        Args:
            query (str): Raw user query.
            mode (str, optional): 'chat', 'task', or None.

        Returns:
            str: Query text prefixed with mode-specific instructions.
        """
        if mode == "chat":
            prefix = "Conversational mode: answer in a warm, helpful, conversational tone. "
        elif mode == "task":
            prefix = "Task mode: answer in a precise, rigorous academic tone. "
        else:
            prefix = ""
        return prefix + query

    def __call__(self, query: str, use_rag: bool = True, mode: str | None = None, injected_context: str | None = None):
        """
        Generate a coaching response.

        Args:
            query (str): User question or draft text.
            use_rag (bool, optional): Whether to retrieve supporting context. Defaults to True.
            mode (str, optional): Interaction mode ('chat' or 'task'). Defaults to None.

        Returns:
            dict: Response package containing:
                - response (str): Cleaned model response.
                - raw_response (str): Raw unprocessed output.
                - retrieve_scores (List[float]): Relevance scores of retrieved docs.
                - used_context (str): Context passed to the LLM.
                - sources (List[dict]): List of retrieved sources.
        """
        context = ""
        retrieve_scores = []
        sources = []

        if injected_context:
            # Use provided context directly (e.g., from attached PDFs)
            context = injected_context
            retrieve_scores = []
            sources = [{"id": 0, "score": 1.0, "text": injected_context[:200], "passed_threshold": True}]
        elif use_rag:
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


# Singleton instance
coach_singleton = CoachingService()