"""
main.py
-------

Entry point for the CoSMIC Coaching Writer API service.

Key responsibilities:
- Initialize FastAPI application.
- Register API routes.
- Provide a root health/info endpoint.
- On startup, ingest academic PDFs from the configured folder (default: ./academic-texts).
"""

import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .api.routes import router
from .core.config import settings
from .services.vector_database import vector_db_singleton


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager.

    Executes startup/shutdown logic for the FastAPI app:
    - On startup: ingest academic PDFs into the vector database.
    - On shutdown: (reserved for cleanup if needed).
    """
    folder = settings.academic_texts_dir
    if os.path.exists(folder):
        try:
            vector_db_singleton.add_pdf_directory(folder)
            print(f"[startup] Loaded PDFs from {folder}")
        except Exception as e:
            print(f"[startup] Failed to ingest PDFs: {e}")
    else:
        print(f"[startup] No academic-texts folder found at {folder}")

    yield  # Startup done; hand control back to FastAPI

    # Shutdown logic could go here (if needed later)
    print("[shutdown] Lifespan cleanup complete.")


# Initialize app with lifespan handler
app = FastAPI(
    title="CoSMIC Coaching Writer",
    version="0.1.0",
    lifespan=lifespan,
)

# Register routes
app.include_router(router)


@app.get("/")
def root():
    """
    Root health/info endpoint.

    Returns:
        dict: Basic service metadata.
    """
    return {
        "service": "CoSMIC-CoachingWriter",
        "message": "Academic writing coaching service.",
        "docs": "/docs",
    }