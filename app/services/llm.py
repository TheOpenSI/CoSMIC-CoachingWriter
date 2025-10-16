"""
llm.py
------

Defines the **LLM interface layer** for the CoSMIC Coaching Writer.

This service wraps an Ollama model (e.g., Qwen or Mistral) and ensures that
prompts, behavioral rules, and ethical constraints are consistently applied
to every generation call.

### Responsibilities
- Construct system and user messages with embedded instructional scaffolds.
- Manage inference calls to the Ollama REST API.
- Enforce “no rewrite” rules through both prompt engineering and post-filtering.
- Fallback gracefully between JSON and NDJSON streaming responses.

### Behavioral contract
The model acts strictly as a **writing coach**, not a writer:
- It provides conceptual feedback, explanations, and illustrative examples.
- It never rewrites, paraphrases, or completes the user’s text.
- Responses must follow the **five-part structure**:
  1. Acknowledge
  2. Analyze
  3. Explain
  4. Illustrate
  5. Encourage

### Configuration
Uses global `settings` for:
- `llm_name`
- `ollama_host`
- `max_new_tokens`
"""

import requests
import json
from typing import List, Dict
from .base import ServiceBase
from ..core.config import settings
from ollama import Client
from .OllamaPullManager import OllamaPullManager


class OllamaChat(ServiceBase):
    """
    Wrapper for an Ollama-based LLM with /api/generate→/api/chat fallback.
    """

    def __init__(self, llm_name: str | None = None, max_new_tokens: int | None = None):
        super().__init__()
        self.llm_name = (llm_name or settings.llm_name).replace("ollama:", "")
        self.host = settings.ollama_host.rstrip("/")
        self.max_new_tokens = max_new_tokens or settings.max_new_tokens

    # ---------- Simplified System Prompt ----------
    SYSTEM_PROMPT = (
        "You are CoSMIC Academic Writing Coach. Guide users to improve their "
        "academic writing without rewriting or completing it.\n"
        "• Offer feedback and conceptual advice only.\n"
        "• Never produce new sentences or rephrase user text.\n"
        "• Focus on clarity, structure, tone, and argument strength.\n"
        "• Teach writing principles briefly and generally.\n"
        "• Stay factual, discipline-neutral, and encouraging.\n"
        "• Respond in five parts: Acknowledge, Analyze, Explain, Illustrate, Encourage.\n"
        "• Keep examples abstract, not tied to user wording.\n"
        "• Maintain concise bullet-style feedback.\n"
        "• Always remind: you guide learning, not write for them."
    )

    def build_messages(self, question: str, context: str = "") -> List[Dict[str, str]]:
        """Build standard message list for chat-type models."""
        system_instruction = self.SYSTEM_PROMPT
        if context:
            system_instruction += f"\n\nContext:\n{context[:3000]}"
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question},
        ]

    # ---------- Inference Call ----------
    def run(self, prompt: str) -> str:
        """Send prompt to Ollama, auto-detect /api/chat vs /api/generate, handle NDJSON, and strip 'thinking'."""
        # ✅ Models that require /api/chat
        chat_only_models = ["qwen", "chat", "phi"]

        model_lower = self.llm_name.lower()
        use_chat = any(tag in model_lower for tag in chat_only_models)

        def _post_chat():
            body = {"model": self.llm_name, "messages": [{"role": "user", "content": prompt}]}
            return requests.post(f"{self.host}/api/chat", json=body, stream=True, timeout=600)

        def _post_generate():
            payload = {"model": self.llm_name, "prompt": prompt}
            return requests.post(f"{self.host}/api/generate", json=payload, stream=True, timeout=600)

        try:
            # 👇 choose based on model name
            r = _post_chat() if use_chat else _post_generate()
            # fallback if first endpoint 404s
            if r.status_code == 404:
                r = _post_generate() if use_chat else _post_chat()
            if r.status_code >= 500:
                raise RuntimeError(f"Ollama backend error {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        text_parts = []
        content_type = r.headers.get("content-type", "")

        if "application/x-ndjson" in content_type or "text/event-stream" in content_type:
            for line in r.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 🚫 Skip model reasoning
                if "thinking" in data:
                    continue
                if "response" in data and data["response"]:
                    text_parts.append(data["response"])
                elif "message" in data and isinstance(data["message"], dict):
                    text_parts.append(data["message"].get("content", ""))
        else:
            try:
                data = r.json()
                if isinstance(data, dict):
                    if "response" in data and data["response"]:
                        text_parts.append(data["response"])
                    elif "message" in data and isinstance(data["message"], dict):
                        text_parts.append(data["message"].get("content", ""))
                elif isinstance(data, list):
                    text_parts.extend(chunk.get("response", "") for chunk in data if isinstance(chunk, dict))
            except Exception:
                text_parts.append(r.text)

        return "".join(text_parts).strip()

    # ---------- Call Operator ----------
    def __call__(self, question: str, context: str = ""):
        """
        Generate a feedback response from the LLM while enforcing no-rewrite rule.
        """
        # Compose prompt
        if context:
            composed = f"{self.SYSTEM_PROMPT}\n\nContext:\n{context[:3000]}\n\nUser: {question}\nAssistant:"
        else:
            composed = f"{self.SYSTEM_PROMPT}\n\nUser: {question}\nAssistant:"

        text = self.run(composed)

        # --- Anti-rewrite filter ---
        forbidden = [
            "you could rephrase", "rewrite this as", "consider writing",
            "for example, you could write", "here’s how it could look", "one possible revision",
        ]
        if any(p in text.lower() for p in forbidden):
            text += (
                "\n\n[Reminder: As your coach, I do not rewrite text. "
                "Apply concepts to your own writing instead.]"
            )
        return text, text


class ManagedOllamaChat(OllamaChat):
    """Adds automatic model pull/verification via OllamaPullManager."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ollama_client = Client(host=self.host)
        self.pull_manager = OllamaPullManager(
            model_name=self.llm_name,
            mode="stochastic",
            interventions=[85, 95],
            max_retries=3,
            fall_back_interval=60,
            ollama_client=self.ollama_client,
        )
        self.pull_manager.pull_model()

    def __call__(self, question: str, context: str = ""):
        self.pull_manager.pull_model()
        return super().__call__(question, context)


# Singleton instance
llm_singleton = ManagedOllamaChat()