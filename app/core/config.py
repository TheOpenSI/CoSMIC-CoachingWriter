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
import torch
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
        """Initialize settings (environment + GPU auto-detect)."""
        load_dotenv()

        # --- Default values ---
        self.llm_name = "ollama:gemma2:2b"
        self.ollama_host = "http://ollama:11434"
        self.vector_db_path = "database/vector_db"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.retrieve_topk = 5
        self.retrieve_score_threshold = 0.55
        self.vector_db_update_threshold = 0.85
        self.device = "cpu"
        self.max_new_tokens = 512
        self.coach_style = "academic"
        self.academic_texts_dir = os.path.join(os.getcwd(), "academic-texts")

        # --- Override with environment variables ---
        self._load_env_overrides()

        # --- CoSMIC-style GPU auto-detection ---
        self.device = self._detect_device()
        print(f"[config] Using device: {self.device}")

    # ----------------------------------------------------------------------
    def _load_env_overrides(self):
        """Load environment variable overrides."""
        self.llm_name = os.getenv("LLM_NAME", self.llm_name)
        self.ollama_host = os.getenv("OLLAMA_HOST", self.ollama_host)
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", self.vector_db_path)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.retrieve_topk = int(os.getenv("RETRIEVE_TOPK", self.retrieve_topk))
        self.retrieve_score_threshold = float(
            os.getenv("RETRIEVE_SCORE_THRESHOLD", self.retrieve_score_threshold)
        )
        self.vector_db_update_threshold = float(
            os.getenv("VECTOR_DB_UPDATE_THRESHOLD", self.vector_db_update_threshold)
        )
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", self.max_new_tokens))
        self.coach_style = os.getenv("COACH_STYLE", self.coach_style)

        # Manual override for device (takes precedence)
        manual_device = os.getenv("DEVICE")
        if manual_device:
            self.device = manual_device.lower()

    def _detect_device(self) -> str:
        """Detect best available compute device (CoSMIC pattern)."""
        # If user explicitly set DEVICE, respect it
        if os.getenv("DEVICE"):
            return os.getenv("DEVICE").lower()

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

# Singleton settings object
settings = Settings()