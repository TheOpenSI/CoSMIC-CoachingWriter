# CoSMIC-CoachingWriter Dockerfile
FROM python:3.11.12-slim

WORKDIR /app

# System deps (optional) for unstructured/PDF improvements
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1 \
    UVICORN_PORT=8001 \
    OLLAMA_HOST=http://ollama:11434

EXPOSE 8001

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
