import os, requests
from typing import List

class Pipeline:
    def __init__(self):
        self.id = "coaching_writer_pipeline"
        self.name = "CoSMIC Coaching Writer"
        self.base = os.getenv("OPENSI_COSMIC_API_BASE_URL", "http://coaching-writer:8001")
        self.max_q = int(os.getenv("MAX_QUERIES_PER_USER", "25"))
        self.user_queries = {}

    async def on_startup(self):
        print("[Pipeline] Coaching Writer ready")

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        if user_message.startswith("###"): return ""
        uid = body.get("user", {}).get("id", "anon")
        role = body.get("user", {}).get("role", "user")
        count = self.user_queries.get(uid, 0)
        if role != "admin" and count >= self.max_q:
            return "Query limit reached."
        self.user_queries[uid] = count + 1

        use_rag = True
        mode = None
        text = user_message.strip()
        if text.lower().startswith('/norag '):
            use_rag = False
            text = text[7:].strip()
        if text.lower().startswith('/mode:'):
            parts = text.split(' ', 1)
            if len(parts) == 2:
                mode = parts[0].split(':',1)[1]
                text = parts[1]

        payload = {"query": text, "use_rag": use_rag, "mode": mode}
        try:
            r = requests.post(f"{self.base}/coach/query", json=payload, timeout=120)
            if r.status_code != 200:
                return f"[Error {r.status_code}] {r.text}"
            data = r.json()
            resp = data.get('response', '')
            if mode:
                resp = f"(Mode: {mode})\n" + resp
            return resp
        except Exception as e:
            return f"[Pipeline Exception] {e}"
