#!/usr/bin/env bash
set -e

# -----------------------------------------------------------------------------
# ollama-entrypoint.sh
# -----------------------------------------------------------------------------
# Purpose:
#   Container startup script for Ollama model service.
# Responsibilities:
#   1. Start Ollama server in the background.
#   2. Wait until API becomes responsive.
#   3. Ensure required model (gemma2:2b) is downloaded.
#   4. Wait up to 10 minutes for completion, then attach.
# -----------------------------------------------------------------------------


MODEL_NAME="gemma2:2b"

echo "[ollama-entrypoint] Starting Ollama server..."
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

echo "[ollama-entrypoint] Ollama API is ready."

# Check if model is already available
if ollama list | grep -q "^${MODEL_NAME}"; then
  echo "[ollama-entrypoint] Model ${MODEL_NAME} already present."
else
  echo "[ollama-entrypoint] Pulling model ${MODEL_NAME}..."
  ollama pull "${MODEL_NAME}" || true

  # Wait up to ~10 minutes for the model to appear
  echo "[ollama-entrypoint] Waiting for model ${MODEL_NAME} to finish downloading..."
  tries=0
  until ollama list | grep -q "^${MODEL_NAME}"; do
    tries=$((tries+1))
    if [ "$tries" -gt 120 ]; then  # 120 * 5s = 600s = 10 minutes
      echo "[ollama-entrypoint] Model ${MODEL_NAME} not ready after 10 minutes" >&2
      exit 1
    fi
    sleep 5
  done
fi

echo "[ollama-entrypoint] Model ${MODEL_NAME} is ready. Attaching to server..."
wait $PID