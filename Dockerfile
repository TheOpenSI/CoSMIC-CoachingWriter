# -----------------------------------------------------------------------------
# Dockerfile — CoSMIC Coaching Writer
# -----------------------------------------------------------------------------
# Purpose:
#   Build the container image for the CoSMIC Coaching Writer FastAPI service.
#   This service provides retrieval-augmented academic writing feedback,
#   integrating with Ollama (for local LLM inference) and OpenWebUI.
#
# Overview:
#   - Based on Python 3.11 slim image for lightweight runtime.
#   - Installs minimal system deps (OCR, PDF parsing).
#   - Copies and installs Python dependencies.
#   - Configures runtime environment and entrypoint.
# -----------------------------------------------------------------------------

FROM python:3.11.12-slim

# Set working directory inside container
WORKDIR /app

# -------------------------------------------------------------------------
# Install required system dependencies:
# - poppler-utils → enables PDF text extraction (via pdftotext)
# - tesseract-ocr → OCR fallback for scanned PDFs
# - libmagic1 → required by python-magic (file type detection)
# -------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . /app

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    UVICORN_PORT=8001 \
    OLLAMA_HOST=http://ollama:11434

# Expose service port for FastAPI
EXPOSE 8001

# -------------------------------------------------------------------------
# Entrypoint configuration:
# - Copies shell script to manage startup
# - Ensures it's executable
# - Starts the FastAPI app via uvicorn
# -------------------------------------------------------------------------
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]