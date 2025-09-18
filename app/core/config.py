import os
from box import Box
import yaml
from dotenv import load_dotenv


class Settings:
    def __init__(self):
        load_dotenv(override=False)
        self.llm_name = os.getenv("LLM_NAME", "ollama:gemma2:2b")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "database/vector_db")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.retrieve_topk = int(os.getenv("RETRIEVE_TOPK", "5"))
        self.retrieve_score_threshold = float(os.getenv("RETRIEVE_SCORE_THRESHOLD", "0.7"))
        self.vector_db_update_threshold = float(os.getenv("VECTOR_DB_UPDATE_THRESHOLD", "0.98"))
        self.device = os.getenv("DEVICE", "cpu")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        self.coach_style = os.getenv("COACH_STYLE", "supportive, concise, academically rigorous")

        config_path = os.path.join(os.getcwd(), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = Box(yaml.safe_load(f) or {})
            self.llm_name = os.getenv("LLM_NAME", data.get("llm_name", self.llm_name))
            self.retrieve_topk = int(os.getenv("RETRIEVE_TOPK", data.get("rag", {}).get("topk", self.retrieve_topk)))
            self.retrieve_score_threshold = float(os.getenv("RETRIEVE_SCORE_THRESHOLD", data.get("rag", {}).get("retrieve_score_threshold", self.retrieve_score_threshold)))
            self.vector_db_path = os.getenv("VECTOR_DB_PATH", data.get("rag", {}).get("vector_db_path", self.vector_db_path))
            self.vector_db_update_threshold = float(os.getenv("VECTOR_DB_UPDATE_THRESHOLD", data.get("rag", {}).get("update_threshold", self.vector_db_update_threshold)))
            self.embedding_model = os.getenv("EMBEDDING_MODEL", data.get("embedding_model", self.embedding_model))
            self.device = os.getenv("DEVICE", data.get("device", self.device))
            coach_conf = data.get("coach", {})
            self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", coach_conf.get("max_new_tokens", self.max_new_tokens)))
            self.coach_style = os.getenv("COACH_STYLE", coach_conf.get("style", self.coach_style))


settings = Settings()
