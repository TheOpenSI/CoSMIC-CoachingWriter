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
import re
from .base import ServiceBase
from .rag import RAG
from .vector_database import get_vector_db
from .llm import ManagedOllamaChat as LLM
import logging

logger = logging.getLogger("coaching_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


class CoachingService(ServiceBase):
    """
    Orchestrates retrieval and LLM calls for coaching responses.

    Attributes:
        llm (OllamaChat): LLM service wrapper.
        rag (RAG): Retrieval service wrapper.
    """

    def __init__(self, user_id: str | None = None):
        super().__init__()
        self.user_id = user_id
        self.llm = LLM()

        logger.debug(f"[INIT] CoachingService created for user_id={user_id!r}")

        # Separate RAGs for system (academic) and per-user (uploads)
        self.system_rag = RAG(get_vector_db("system"))
        self.user_rag = RAG(get_vector_db(user_id)) if user_id else None

        logger.debug(
            f"[INIT] System RAG path: {self.system_rag.db.local_database_path if hasattr(self.system_rag, 'db') else 'unknown'}"
        )
        if self.user_rag:
            logger.debug(
                f"[INIT] User RAG path: {self.user_rag.db.local_database_path if hasattr(self.user_rag, 'db') else 'unknown'}"
            )

    def build_prompt(self, query: str, mode: str | None = None):
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
        injected_filename: str | None = None
    ):
        """
        Generate a CoSMIC-style coaching response using system RAG + uploaded user text.
        """
        logger.debug(f"[CALL] User={self.user_id!r}, Mode={mode}, use_rag={use_rag}, Injected={bool(injected_context)}")
        
        context = ""
        retrieve_scores = []
        sources = []

        # --- Step 1: Get system academic context ---
        if use_rag:
        # --- System RAG ---
            sys_ctx, sys_scores, sys_sources = self.system_rag(query)
            logger.debug(f"[RAG] System retrieved {len(sys_sources)} sources")
            retrieve_scores.extend(sys_scores)
            sources.extend(sys_sources)
            context = sys_ctx or ""

            # --- User RAG ---
            if self.user_rag:
                usr_ctx, usr_scores, usr_sources = self.user_rag(query)
                logger.debug(f"[RAG] User retrieved {len(usr_sources)} sources")
                if usr_ctx:
                    logger.debug(f"[RAG] Merging user context snippet (len={len(usr_ctx)})")
                    context = (
                        f"Here is the user's uploaded material for personalized feedback:\n"
                        f"{usr_ctx[:1000]}\n\n"
                        f"Use the following academic writing references:\n"
                        f"{context}"
                    )
                    retrieve_scores.extend(usr_scores)
                    sources.extend(usr_sources)
                else:
                    logger.debug("[RAG] No user-specific context found")

        # --- Injected context ---
        if injected_context:
            logger.debug(f"[CTX] Injected context present (len={len(injected_context)})")
            context = (
                f"Here is the user's uploaded text for analysis:\n"
                f"{injected_context[:1000]}\n\n"
                f"Use the following academic writing references to guide your feedback:\n"
                f"{context}"
            )

        # --- Generate ---
        prompt = self.build_prompt(query, mode)
        logger.debug(f"[LLM] Sending prompt len={len(prompt)}, context len={len(context)}")

        response, raw = self.llm(prompt, context=context)

        logger.debug(f"[LLM] Response len={len(response)}, Sources={len(sources)}")

        # --- Step 4: Clean up headings ---
        heading_patterns = [
            r"(?im)^\s*Acknowledge\s*[:\-]?\s*",
            r"(?im)^\s*Analyze\s*[:\-]?\s*",
            r"(?im)^\s*Explain\s*[:\-]?\s*",
            r"(?im)^\s*Illustrate\s*[:\-]?\s*",
            r"(?im)^\s*Encourage\s*[:\-]?\s*",
        ]
        for pattern in heading_patterns:
            response = re.sub(pattern, "", response)

        # --- Step 5: Append a sources legend ---
        if sources:
            unique_names = []
            for s in sources:
                name = (
                    s.get("source")
                    or s.get("metadata", {}).get("source")
                    or s.get("path")
                    or "context"
                )
                base = os.path.basename(name)
                if base not in unique_names:
                    unique_names.append(base)
            legend_lines = [f"[{i}] {n}" for i, n in enumerate(unique_names, start=1)]
            response += f"\n\nSources:\n" + "\n".join(legend_lines)

        # --- Step 6: Normalize citation numbers ---
        if sources:
            max_id = len(unique_names)
            response = re.sub(r'\[(\d+)\]', lambda m: f"[{min(int(m.group(1)), max_id)}]", response)

        logger.debug(f"[RETURN] User={self.user_id!r}, context hash={hash(context)}, sources={[os.path.basename(s.get('source', '?')) for s in sources]}")

        return {
            "response": response,
            "raw_response": raw,
            "retrieve_scores": retrieve_scores,
            "used_context": context,
            "sources": sources,
        }

