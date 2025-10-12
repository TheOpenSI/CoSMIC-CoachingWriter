"""
main.py
-------

Entry point for the **CoSMIC Coaching Writer** FastAPI service.

Responsibilities:
  • Initialize FastAPI app and register API routes.
  • Load academic reference PDFs into the vector database on startup.
  • Verify the Ollama model is present and ready.
  • Expose health and metadata endpoints.

Startup Sequence:
  1. Scan the `academic-texts/` directory and ingest all PDFs.
  2. Initialize FAISS vector database (via `vector_db_singleton`).
  3. Verify that the configured LLM model is pulled and ready in Ollama.
  4. Serve routes defined in `app/api/routes.py`.
"""

import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .api.routes import router
from .core.config import settings
from .services.vector_database import vector_db_singleton
from ollama import Client
from .services.OllamaPullManager import OllamaPullManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for startup/shutdown events.

    During startup:
      - Loads all reference PDFs from `academic-texts/` into FAISS.
      - Verifies or downloads the Ollama model.

    On shutdown:
      - Reserved for future cleanup tasks.
    """
    folder = settings.academic_texts_dir
    if os.path.exists(folder):
        try:
            vector_db_singleton.add_document_directory(folder)
            print(f"[startup] Loaded PDFs from {folder}")
        except Exception as e:
            print(f"[startup] Failed to ingest PDFs: {e}")
    else:
        print(f"[startup] No academic-texts folder found at {folder}")

    # Ensure Ollama model availability
    try:
        client = Client(host=settings.ollama_host)
        pull_manager = OllamaPullManager(
            model_name=settings.llm_name.replace("ollama:", ""),
            mode="stochastic",
            interventions=[85, 95],
            max_retries=3,
            fall_back_interval=60,
            ollama_client=client,
        )
        pull_manager.pull_model()
        print(f"[startup] Ollama model {settings.llm_name} verified and ready.")
    except Exception as e:
        print(f"[startup] Failed to verify/pull model: {e}")

    yield
    print("[shutdown] Lifespan cleanup complete.")

# Create and configure FastAPI app
app = FastAPI(
    title="CoSMIC Coaching Writer",
    version="0.1.0",
    lifespan=lifespan,
)

# Register all routes
app.include_router(router)

@app.get("/")
def root():
    """Health/info endpoint for system diagnostics."""
    return {
        "service": "CoSMIC-CoachingWriter",
        "message": "Academic writing coaching service.",
        "docs": "/docs",
    }