#!/usr/bin/env bash
set -euo pipefail

if [ -f /app/.env ]; then
  echo "[entrypoint] Loading .env"
  set -a; source /app/.env; set +a
fi

echo "[entrypoint] Starting Uvicorn on :8001 (reload=${RELOAD:-false})"

if [ "${RELOAD:-false}" = "true" ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
else
  exec uvicorn app.main:app --host 0.0.0.0 --port 8001
fi
