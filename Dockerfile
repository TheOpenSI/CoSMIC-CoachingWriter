# Dockerfile for CoSMIC Coaching Writer
# -------------------------------------
# Builds a container image for the FastAPI-based Coaching Writer service.
# Includes Python dependencies, application source code, and runtime config.

FROM python:3.11.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

ENV PYTHONUNBUFFERED=1 \
    UVICORN_PORT=8001 \
    OLLAMA_HOST=http://ollama:11434
# Expose FastAPI default port
EXPOSE 8001

# Default entrypoint: run uvicorn with auto-reload disabled
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
