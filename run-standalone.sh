#!/usr/bin/env bash
set -euo pipefail


# run-standalone.sh
# -----------------
# Convenience script to build and run the full stack (via Docker Compose).
# Responsibilities:
# - Launch containers: Ollama, Coaching Writer, Pipelines, OpenWebUI.
# - Wait briefly for services to initialize.
# - Perform a health check on the Coaching Writer service.
# - Print access URL for OpenWebUI.

COMPOSE_FILE=docker-compose.yaml

echo "[standalone] Building & starting stack (ollama + coaching-writer + pipelines + open-webui)"
docker compose -f ${COMPOSE_FILE} up -d --build

echo "[standalone] Waiting 5s for services to settle..."
sleep 5

echo "[standalone] Health check coaching-writer"
curl -s http://localhost:8001/health || echo "(health endpoint not ready yet)"

echo "[standalone] OpenWebUI: http://localhost:8080"
