#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# entrypoint.sh — Coaching Writer FastAPI Entrypoint
# -----------------------------------------------------------------------------
# Responsibilities:
#   - Load environment variables (if `.env` exists).
#   - Reset file_list.csv to keep vector DB fresh.
#   - Start Uvicorn server on port 8001.
#   - Optional live reload in dev mode.
# -----------------------------------------------------------------------------


if [ -f /app/.env ]; then
  echo "[entrypoint] Loading .env"
  set -a; source /app/.env; set +a
fi

mkdir -p /app/database/vector_db
echo "Source,Time,Comment" > /app/database/vector_db/file_list.csv
echo "[entrypoint] Reset file_list.csv"

echo "[entrypoint] Starting Uvicorn on :8001 (reload=${RELOAD:-false})"

if [ "${RELOAD:-false}" = "true" ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
else
  exec uvicorn app.main:app --host 0.0.0.0 --port 8001
fi
