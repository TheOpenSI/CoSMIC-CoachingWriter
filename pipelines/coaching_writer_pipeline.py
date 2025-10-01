"""
coaching_writer_pipeline.py
---------------------------

This module defines the OpenWebUI pipeline for the **CoSMIC Coaching Writer**.
Pipelines act as connectors between user input (chat messages, uploaded files)
and the backend service (`/coach/query`) that performs retrieval-augmented
generation (RAG) and coaching.

Key responsibilities:
- Detect conversational vs. task-based queries.
- Enforce safety rules (no rewriting or text generation for the user).
- Manage per-user query limits.
- Forward requests to the coaching backend with the appropriate payload.
"""

import os
import requests
from typing import List

# --- Mode Detection Helper ---
def detect_mode(text: str) -> str:
    """
    Automatically detect whether the user message is a conversational query
    (e.g., greetings, tool usage) or a task-related academic coaching query
    (e.g., assignments, essays, drafts).

    Returns:
        str: 'chat', 'task', or None if no mode detected.
    """
    chat_keywords = ["hi", "hello", "thanks", "how do i", "what is this", "help me use"]
    task_keywords = ["assignment", "essay", "draft", "section", "paper", "review", "improve"]

    lower = text.lower()
    if any(kw in lower for kw in task_keywords):
        return "task"
    if any(kw in lower for kw in chat_keywords):
        return "chat"
    return None


# --- Safety Instruction ---
SAFETY_INSTRUCTION = (
    "IMPORTANT: Never rewrite or generate new text for the user. "
    "Only provide constructive feedback, suggestions, and advice "
    "on how the user can improve their own writing."
)


# --- Pipeline Class ---
class Pipeline:
    """
    OpenWebUI pipeline for the CoSMIC Coaching Writer.

    Attributes:
        id (str): Unique pipeline identifier.
        name (str): Human-readable pipeline name.
        base (str): Base URL of the coaching backend service.
        max_q (int): Maximum queries allowed per user (non-admin).
        user_queries (dict): Tracks query counts per user.
    """

    def __init__(self):
        self.id = "coaching_writer_pipeline"
        self.name = "CoSMIC Coaching Writer"
        self.base = os.getenv("OPENSI_COSMIC_API_BASE_URL", "http://coaching-writer:8001")
        self.max_q = int(os.getenv("MAX_QUERIES_PER_USER", "25"))
        self.user_queries = {}

    async def on_startup(self):
        """
        Lifecycle hook executed when the pipeline starts up.
        Prints a readiness message.
        """
        print("[Pipeline] Coaching Writer ready")

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        """
        Main entry point for processing user queries.

        Args:
            user_message (str): Raw user message text.
            model_id (str): Identifier for the LLM model in use.
            messages (List[dict]): Conversation history (ignored in this pipeline).
            body (dict): Additional metadata including user info and uploaded files.

        Returns:
            str: Response text from the backend or error message.
        """
        # Ignore system messages
        if user_message.startswith("###"):
            return ""

        uid = body.get("user", {}).get("id", "anon")
        role = body.get("user", {}).get("role", "user")
        documents = body.get("files") or body.get("documents") or []

        # Enforce per-user query limits
        count = self.user_queries.get(uid, 0)
        if role != "admin" and count >= self.max_q:
            return "Query limit reached."
        self.user_queries[uid] = count + 1

        use_rag = True
        mode = None
        text = user_message.strip()

        # --- Special commands ---
        if text.lower().startswith('/norag '):
            use_rag = False
            text = text[7:].strip()
        if text.lower().startswith('/mode:'):
            parts = text.split(' ', 1)
            if len(parts) == 2:
                mode = parts[0].split(':', 1)[1]
                text = parts[1]

        # --- Auto-detect mode ---
        if not mode:
            mode = detect_mode(text)

        # --- Append safety instruction ---
        query_with_safety = f"{text}\n\n{SAFETY_INSTRUCTION}"

        payload = {
            "query": query_with_safety,
            "use_rag": use_rag,
            "mode": mode,
            "documents": documents
        }

        try:
            r = requests.post(f"{self.base}/coach/query", json=payload, timeout=120)
            if r.status_code != 200:
                return f"[Error {r.status_code}] {r.text}"
            data = r.json()
            resp = data.get('response', '')

            # Prepend mode tag
            if mode:
                resp = f"(Mode: {mode})\n" + resp

            return resp
        except Exception as e:
            return f"[Pipeline Exception] {e}"