#!/usr/bin/env bash
set -euo pipefail

echo "[clean] Removing Python cache directories..."
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type d -name .pytest_cache -prune -exec rm -rf {} +

echo "[clean] Optional: remove local vector DB & uploads (commented out by default)"
# rm -rf database/vector_db || true
# rm -rf uploads || true

if command -v docker >/dev/null 2>&1; then
  echo "[clean] (Optional) prune dangling Docker images (uncomment to use)"
  # docker image prune -f
  echo "[clean] (Optional) prune build cache (uncomment to use)"
  # docker builder prune -f
fi

echo "[clean] Done."
