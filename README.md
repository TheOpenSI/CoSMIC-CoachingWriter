# OpenSI-CoSMIC - CochingWriter

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/ACIS-2025-oliver.svg)](https://arxiv.org/abs/2408.04910)
[![python](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-2025-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![RAG Ollama](https://img.shields.io/badge/RAG-Ollama-000000.svg?style=flat&logo=ollama&logoColor=white)](https://ollama.ai)
[![DebianBadge](https://badges.debian.net/badges/debian/stable/docker/version.svg)](https://www.docker.com/)

This is the official implementation of the Open Source Institute-CoSMIC-CoachingWriter v1.0.0.


# CoSMIC-CoachingWriter

Microservice in the CoSMIC ecosystem that provides an academic writing coaching assistant using a Small Language Model (SLM) (default: `gemma2:2b` via Ollama) with Retrieval-Augmented Generation (RAG).

## Key Features

* Upload academic PDFs to build a private vector knowledge base.
* RAG-enabled coaching: context-aware suggestions, critique, structure advice.
* Deterministic, low-temperature responses for clarity and rigor.
* Lightweight Ollama-based inference (no large GPU mandatory; CPU viable for small models).
* Simple REST API (FastAPI) for integration with OpenWebUI / other CoSMIC services.

## Architecture

```
						+-----------------------------+
						|    OpenWebUI / Client       |
						+---------------+-------------+
														|
														| REST (upload/query)
														v
 +---------------------------------------------------+
 |            CoSMIC-CoachingWriter Service          |
 |                                                   |
 |  +-----------+    +---------+    +--------------+ |
 |  |  Routes   +--> | Coach   +--> |   Ollama LLM | |
 |  +-----------+    +----+----+    +------+-------+ |
 |                        |                 ^        |
 |                        v                 |        |
 |                 +-------------+   RAG    |        |
 |                 |   RAG       +----------+        |
 |                 +------+------+                    |
 |                        |                           |
 |                 +------+------+                    |
 |                 | Vector DB | (FAISS + Embeddings) |
 |                 +-------------+                    |
 +---------------------------------------------------+
```

### Components

| Component | Purpose |
|-----------|---------|
| `app/services/llm.py` | Minimal Ollama wrapper; consolidated responses for internal calls. |
| `app/services/vector_database.py` | FAISS-based vector store for PDF + text ingestion. |
| `app/services/rag.py` | Retrieves top-k relevant chunks and builds compact context. |
| `app/services/coach.py` | Orchestrates RAG + LLM to produce coached feedback. |
| `app/api/routes.py` | REST endpoints (FastAPI). |

## Endpoints

Primary REST endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service + model status. |
| POST | `/documents/upload` | Upload a PDF (multipart/form-data). |
| POST | `/documents/ingest-text` | Add raw text to vector DB. |
| GET | `/documents` | List ingested sources (from catalogue). |
| POST | `/coach/query` | Body: `{query, use_rag, mode}` → coached answer.

OpenAI-compatible minimal endpoints:

| Method | Path | Notes |
|--------|------|-------|
| GET | `/v1/models` | Lists the backend model id. |
| POST | `/v1/chat/completions` | Accepts standard messages array. Prefix `/norag` in user content to disable retrieval.

Ollama-compatible minimal endpoints (for UI compatibility): `/api/version`, `/api/tags`, `/api/show`, `/api/pull`, `/api/ps`, `/api/delete`, `/api/generate`, `/api/chat`.

`mode` (optional) supports hints like `critique`, `improve`, `structure`, `abstract`, `references`.

## Quick Start

This compose file starts everything you need: Ollama (the model server), the CoachingWriter API, and OpenWebUI.

1) Start everything

```bash
docker compose up -d --build
```

2) Check it’s running

```bash
curl -s http://localhost:8001/health | jq
```

3) Ask the coach a question

```bash
curl -s -X POST http://localhost:8001/coach/query \
	-H 'Content-Type: application/json' \
	-d '{"query":"Give two suggestions to improve this abstract.", "use_rag": true}' | jq
```

4) Open the website

Open http://localhost:8080 in your browser.

Optional: start the small “pipelines” helper (not required for normal use):

```bash
docker compose --profile pipelines up -d --build pipelines
```

### Simple API examples

- OpenAI-style chat (works with many clients):

```bash
curl -s http://localhost:8001/v1/models | jq
curl -s -X POST http://localhost:8001/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{"model":"gemma2:2b","messages":[{"role":"user","content":"Give two suggestions to improve this abstract."}]}' | jq
```

Tip: to turn off retrieval for a single prompt, start your message with “/norag”:

```bash
curl -s -X POST http://localhost:8001/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{"model":"gemma2:2b","messages":[{"role":"user","content":"/norag Summarize best practices for literature review clarity."}]}' | jq
```

- Coach endpoint (simple JSON body):

```bash
curl -s -X POST http://localhost:8001/coach/query \
	-H 'Content-Type: application/json' \
	-d '{"query": "Improve coherence of this paragraph about method validity:", "use_rag": true}' | jq
```

- Upload a PDF to use as context:

```bash
curl -s -X POST http://localhost:8001/documents/upload \
	-F 'file=@/path/to/paper.pdf' | jq
```

## Configuration

* `.env` overrides `config.yaml`.
* Key variables: `LLM_NAME`, `VECTOR_DB_PATH`, `EMBEDDING_MODEL`, `RETRIEVE_TOPK`, `RETRIEVE_SCORE_THRESHOLD`, `MAX_NEW_TOKENS`, `COACH_STYLE`.

### RAG Behaviour

* By default, retrieval is enabled. It returns up to `RETRIEVE_TOPK` chunks above `RETRIEVE_SCORE_THRESHOLD` and injects them into the prompt as `[id] snippet` lines.
* The model response may append `Sources: [0], [3] ...` if any chunks were used.
* Disable retrieval per request:
  * Coach REST: set `use_rag: false` in `/coach/query`.
  * OpenAI: prefix the user message with `/norag`.
* OpenAI-style responses include a `cosmic` object with `rag_used`, `source_count`, `underlying_model`, and `sources`.
* Note: `/coach/query` response schema does not include `sources` (it returns `response`, `raw_response`, `retrieve_scores`, `used_context`).

## Tests

```bash
pytest -q
```

## Future Enhancements

* Streaming for OpenAI-compatible endpoint.
* Auth (API key / JWT) & rate limiting.
* Section-aware structural analysis.
* Citation quality assessment & reference normalization.
* Evaluation harness with golden feedback pairs.
* Web RAG retrieval: fetch and chunk web pages as additional context sources (with allowlist/robots handling and basic content cleaning).

## License

Inherit CoSMIC project licensing; ensure compliance with model provider (Gemma / Ollama / Hugging Face) licenses where applicable.


