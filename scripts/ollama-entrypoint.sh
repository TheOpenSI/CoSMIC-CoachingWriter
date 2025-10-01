#!/usr/bin/env sh
set -e

# ollama-entrypoint.sh
# --------------------
# Entrypoint script for the Ollama container.
# Responsibilities:
# - Start the Ollama server in the background.
# - Wait until the Ollama API becomes responsive.
# - Ensure that the required model (gemma2:2b) is pulled and available.
# - Attach to the Ollama server process.

echo "[ollama-entrypoint] Starting ollama server..."
ollama serve &
PID=$!

echo "[ollama-entrypoint] Waiting for Ollama API..."
tries=0
until ollama list >/dev/null 2>&1; do
  tries=$((tries+1))
  if [ "$tries" -gt 60 ]; then
    echo "[ollama-entrypoint] Ollama API did not become ready in time" >&2
    exit 1
  fi
  sleep 2
done

if ! ollama list | grep -q '^gemma2:2b'; then
  echo "[ollama-entrypoint] Pulling model gemma2:2b..."
  ollama pull gemma2:2b || true
fi

echo "[ollama-entrypoint] Model check complete; attaching to server"
wait $PID
