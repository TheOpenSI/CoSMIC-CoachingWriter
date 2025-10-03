#!/usr/bin/env bash
set -euo pipefail

# entrypoint.sh
# -------------
# Entrypoint for the Coaching Writer FastAPI service.
# Responsibilities:
# - Load environment variables from `.env` if present.
# - Start Uvicorn server serving the `app.main:app` application.
# - Support live-reload in development mode (via RELOAD=true).

if [ -f /app/.env ]; then
  echo "[entrypoint] Loading .env"
  set -a; source /app/.env; set +a
fi

# Reset file_list.csv each time the container starts
mkdir -p /app/database/vector_db
echo "Source,Time,Comment" > /app/database/vector_db/file_list.csv
echo "[entrypoint] Reset file_list.csv"

echo "[entrypoint] Starting Uvicorn on :8001 (reload=${RELOAD:-false})"

if [ "${RELOAD:-false}" = "true" ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
else
  exec uvicorn app.main:app --host 0.0.0.0 --port 8001
fi
