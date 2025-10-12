"""
Configuration Loader

This module centralizes configuration for the Coaching Writer service.
It supports multiple sources:
1. Default values (lowest precedence).
2. Environment variables.
3. YAML configuration file (optional, can override defaults).

The resolved configuration is exposed as a singleton `settings` object
that can be imported anywhere in the codebase.
"""

import os
import yaml
from dotenv import load_dotenv


class Settings:
    """
    Application configuration settings.

    Priority order for values:
    1. Environment variables (highest precedence).
    2. YAML config file (if present).
    3. Built-in defaults.

    Attributes:
        llm_name (str): Name/identifier of the LLM (default: "ollama:gemma2:2b").
        ollama_host (str): Base URL of the Ollama service.
        vector_db_path (str): Path where FAISS vector index will be stored.
        embedding_model (str): HuggingFace model used for embeddings.
        retrieve_topk (int): Number of top documents to retrieve for RAG.
        retrieve_score_threshold (float): Minimum similarity score to keep a doc.
        vector_db_update_threshold (float): Threshold for deduplication checks.
        device (str): Device used for embeddings ("cpu" or "cuda").
        max_new_tokens (int): Max tokens for LLM generations.
        coach_style (str): Coaching style/tone (e.g., "academic", "conversational").
    """

    def __init__(self):
        """Initialize settings by loading from environment and config file."""
        load_dotenv()

        # defaults
        self.llm_name = "ollama:qwen3:4b"
        self.ollama_host = "http://ollama:11434"
        self.vector_db_path = "database/vector_db"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.retrieve_topk = 5
        self.retrieve_score_threshold = 0.65
        self.vector_db_update_threshold = 0.85
        self.device = "cpu"
        self.max_new_tokens = 512
        self.coach_style = "academic"
        self.academic_texts_dir: str = os.path.join(os.getcwd(), "academic-texts")

        # environment variable overrides
        self._load_env()

        # optional YAML overrides
        self._load_yaml("config.yaml")

    def _load_env(self):
        """Override defaults with environment variables if present."""
        self.llm_name = os.getenv("LLM_NAME", self.llm_name)
        self.ollama_host = os.getenv("OLLAMA_HOST", self.ollama_host)
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", self.vector_db_path)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.device = os.getenv("DEVICE", self.device)

    def _load_yaml(self, path: str):
        """
        Override defaults with values from a YAML file, if present.

        Args:
            path (str): Path to the YAML file.
        """
        if os.path.exists(path):
            with open(path, "r") as f:
                config = yaml.safe_load(f) or {}
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)


# Singleton settings object
settings = Settings()