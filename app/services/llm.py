"""
llm.py
------

This module defines the LLM service wrapper used by the CoSMIC Coaching Writer.
It interfaces with an Ollama backend to generate responses based on user queries.

Key responsibilities:
- Define system instructions and rules for the academic writing coach.
- Compose prompts with or without retrieval-augmented context.
- Enforce behavioral constraints (never rewrite text, only provide feedback).
- Handle both standard JSON and fallback NDJSON streaming responses.
"""

import requests
from typing import List, Dict
from .base import ServiceBase
from ..core.config import settings
from ollama import Client
from .OllamaPullManager import OllamaPullManager


class OllamaChat(ServiceBase):
    """
    Wrapper for an Ollama-based LLM.

    Attributes:
        llm_name (str): Model name to use.
        host (str): Base host URL for the Ollama server.
        max_new_tokens (int): Maximum tokens to generate.
    """

    def __init__(self, llm_name: str | None = None, max_new_tokens: int | None = None):
        super().__init__()
        self.llm_name = (llm_name or settings.llm_name).replace("ollama:", "")
        self.host = settings.ollama_host.rstrip('/')
        self.max_new_tokens = max_new_tokens or settings.max_new_tokens

    def build_messages(self, question: str, context: str = "") -> List[Dict[str, str]]:
        """
        Construct a list of system + user messages suitable for chat models.

        Args:
            question (str): User question or draft section.
            context (str, optional): Retrieved supporting context. Defaults to "".

        Returns:
            List[Dict[str, str]]: Chat-format message list.
        """
        system_instruction = (
            "You are CoSMIC Academic Writing Coach.\n\n"
            "Rules:\n"
            "- Always give advice and feedback, never rewrite text or generate paragraphs for the user.\n"
            "- When discussing improvements, point out clarity, argument strength, structure, tone, methodology, and use of sources.\n"
            "- Use a friendly, conversational style if the user is just chatting or asking about the tool.\n"
            "- Use a formal, academic style when analyzing or critiquing user-provided texts or assignments.\n"
            "- If Relevant Context is provided, treat it as the user's draft and analyze it directly—do NOT ask the user to paste their text again.\n"
            "- Explicitly remind the user that you do not generate new text, only guidance."
        )
        if context:
            system_instruction += f"\nRelevant Context: {context[:4000]}"

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question}
        ]

    def __call__(self, question: str, context: str = ""):
        """
        Generate a response from the LLM.

        Args:
            question (str): User input text.
            context (str, optional): Additional supporting context. Defaults to "".

        Returns:
            Tuple[str, str]: (cleaned response, raw response).
        """
        sys_prompt = (
            "You are CoSMIC Academic Writing Coach.\n\n"
            "Rules:\n"
            "- Always give advice and feedback, never rewrite text or generate paragraphs for the user.\n"
            "- When discussing improvements, point out clarity, argument strength, structure, tone, methodology, and use of sources.\n"
            "- Use a friendly, conversational style if the user is just chatting or asking about the tool.\n"
            "- Use a formal, academic style when analyzing or critiquing user-provided texts or assignments.\n"
            "- If Relevant Context is provided, treat it as the user's draft and analyze it directly—do NOT ask the user to paste their text again.\n"
            "- Explicitly remind the user that you do not generate new text, only guidance."
        )
        if context:
            sys_prompt += f"\nRelevant Context: {context[:4000]}"

        composed = f"{sys_prompt}\n\nUser: {question}\nAssistant:"

        payload = {
            "model": self.llm_name,
            "prompt": composed,
            "options": {"num_predict": self.max_new_tokens, "temperature": 0.1},
            "stream": False,
        }
        url = f"{self.host}/api/generate"
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()

        try:
            data = r.json()
            if isinstance(data, dict) and 'response' in data:
                return data.get('response') or '', data.get('response') or ''
            elif isinstance(data, list):
                content = ''.join([chunk.get('response', '') for chunk in data if isinstance(chunk, dict)])
                if content:
                    return content, content
        except Exception:
            # Handle NDJSON or raw fallback
            text = r.text
            aggregated = []
            try:
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = __import__('json').loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        msg = obj.get('message', {}) if isinstance(obj.get('message'), dict) else None
                        if isinstance(msg, dict) and 'content' in msg:
                            aggregated.append(msg.get('content') or '')
                        if 'response' in obj:
                            aggregated.append(obj.get('response') or '')
                if aggregated:
                    return ''.join(aggregated), text
            except Exception:
                pass
            return text, text

class ManagedOllamaChat(OllamaChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ollama_client = Client(host=self.host)
        self.pull_manager = OllamaPullManager(
            model_name=self.llm_name,
            mode="stochastic",
            interventions=[85, 95],
            max_retries=3,
            fall_back_interval=60,
            ollama_client=self.ollama_client
        )
        # Pull or verify model at startup
        self.pull_manager.pull_model()

    def __call__(self, question: str, context: str = ""):
        # Recheck availability before each inference
        self.pull_manager.pull_model()
        return super().__call__(question, context)

# Singleton instance
llm_singleton = ManagedOllamaChat()