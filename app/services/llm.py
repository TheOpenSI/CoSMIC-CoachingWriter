import requests
from typing import List, Dict
from .base import ServiceBase
from ..core.config import settings


class OllamaChat(ServiceBase):
    def __init__(self, llm_name: str | None = None, max_new_tokens: int | None = None):
        super().__init__()
        self.llm_name = (llm_name or settings.llm_name).replace("ollama:", "")
        self.host = settings.ollama_host.rstrip('/')
        self.max_new_tokens = max_new_tokens or settings.max_new_tokens

    def build_messages(self, question: str, context: str = "") -> List[Dict[str, str]]:
        system_instruction = (
            "You are CoSMIC Academic Writing Coach. Provide constructive, concise, rigorous feedback. "
            "Return bullet points when possible. If the user provides a draft section, focus on clarity, "
            "argument strength, coherence, methodology precision, literature integration, and academic tone."
        )
        if context:
            system_instruction += f"\nRelevant Context: {context[:4000]}"
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question}
        ]

    def __call__(self, question: str, context: str = ""):
        # Compose a single prompt to use with Ollama's generate endpoint, which cleanly supports
        # non-streaming responses via { stream: false }.
        sys_prompt = (
            "You are CoSMIC Academic Writing Coach. Provide constructive, concise, rigorous feedback. "
            "Return bullet points when possible. If the user provides a draft section, focus on clarity, "
            "argument strength, coherence, methodology precision, literature integration, and academic tone."
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
            if isinstance(data, dict):
                # Non-streaming generate returns a single dict with 'response'
                if 'response' in data:
                    content = data.get('response') or ''
                    return content, content
            elif isinstance(data, list):
                # In case server still returned a list of chunks, concatenate their 'response'
                content = ''.join([chunk.get('response', '') for chunk in data if isinstance(chunk, dict)])
                if content:
                    return content, content
        except Exception:
            # Fallback: attempt to parse NDJSON stream aggregated in text
            text = r.text
            aggregated = []
            try:
                # Handle both newline-delimited JSON and single JSON line
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = __import__('json').loads(line)
                    except Exception:
                        continue
                    # Prefer chat-style message content if present
                    if isinstance(obj, dict):
                        # handle both chat-style and generate-style events
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


llm_singleton = OllamaChat()
