#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=docker-compose.yaml

echo "[standalone] Building & starting stack (ollama + coaching-writer + pipelines + open-webui)"
docker compose -f ${COMPOSE_FILE} up -d --build

echo "[standalone] Waiting 5s for services to settle..."
sleep 5

echo "[standalone] Health check coaching-writer"
curl -s http://localhost:8001/health || echo "(health endpoint not ready yet)"

echo "[standalone] OpenWebUI: http://localhost:8080"
