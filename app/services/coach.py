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

import os
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

    def __call__(
        self,
        query: str,
        use_rag: bool = True,
        mode: str | None = None,
        injected_context: str | None = None,
        injected_filename: str | None = None,   # 👈 NEW
        user_id: str | None = None
    ):
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

        # --- Step 1: Retrieve context ---
        if injected_context and use_rag:
            # Retrieve relevant academic guide context
            rag_context, retrieve_scores, rag_sources = self.rag(query)

            # Add uploaded doc as a distinct source
            context = (
                f"Here is the user's uploaded text for analysis:\n"
                f"{injected_context[:1000]}\n\n"
                f"Use the following academic writing references to guide your feedback:\n"
                f"{rag_context}"
            )

            # Only use academic guide sources (exclude upload)
            sources = rag_sources


        elif injected_context:
            # Only uploaded doc (no RAG)
            context = injected_context
            sources = []

        elif use_rag:
            # Only RAG context (no upload)
            context, retrieve_scores, sources = self.rag(query)

        # --- Step 2: Build prompt ---
        prompt = self.build_prompt(query, mode=mode)

        # --- Step 3: Build numbered context block (group by file) ---
        if sources:
            unique_sources = []
            numbered_context = "Context:\n"

            # Collect one entry per unique filename
            for s in sources:
                name = s.get("source") or s.get("metadata", {}).get("source") or "context"
                if name not in [u["source"] for u in unique_sources]:
                    unique_sources.append({"source": name, "text": s.get("text", "")})

            # Label each unique file as [1], [2], etc.
            for i, src in enumerate(unique_sources, start=1):
                preview = src["text"][:800].strip().replace("\n", " ")
                numbered_context += f"[{i}] ({os.path.basename(src['source'])}) {preview}\n"

            context = numbered_context.strip()

        # --- Step 4: Generate response from LLM ---
        response, raw = self.llm(prompt, context=context)

        heading_patterns = [
            r"(?im)^\s*Acknowledge\s*[:\-]?\s*",
            r"(?im)^\s*Analyze\s*[:\-]?\s*",
            r"(?im)^\s*Explain\s*[:\-]?\s*",
            r"(?im)^\s*Illustrate\s*[:\-]?\s*",
            r"(?im)^\s*Encourage\s*[:\-]?\s*",
        ]

        import re
        for pattern in heading_patterns:
            response = re.sub(pattern, "", response)

        # --- Step 5: Append a full legend (deduplicated filenames) ---
        if sources and use_rag:
            unique_names = []
            for s in sources:
                name = (
                    s.get("source")
                    or s.get("path")
                    or s.get("metadata", {}).get("source")
                    or s.get("name")
                    or s.get("origin")
                    or "context"
                )
                name = os.path.basename(name)
                if name not in unique_names:
                    unique_names.append(name)

            legend_lines = [f"[{i}] {n}" for i, n in enumerate(unique_names, start=1)]
            legend = "\n".join(legend_lines)
            response += f"\n\nSources:\n{legend}"

        # --- Step 6: Normalize in-text citations ---
        if sources and use_rag:
            # Replace any [number] greater than your total unique sources
            max_id = len(unique_names)
            import re
            response = re.sub(r'\[(\d+)\]', 
                            lambda m: f"[{min(int(m.group(1)), max_id)}]", 
                            response)

        return {
            "response": response,
            "raw_response": raw,
            "retrieve_scores": retrieve_scores,
            "used_context": context,
            "sources": sources,
        }


# Singleton instance
coach_singleton = CoachingService()