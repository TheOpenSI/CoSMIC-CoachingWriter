# OpenSI-CoSMIC - Coaching Writer  

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/ACIS-2025-oliver.svg)](https://arxiv.org/abs/2408.04910)
[![python](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-2025-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![RAG Ollama](https://img.shields.io/badge/RAG-Ollama-000000.svg?style=flat&logo=ollama&logoColor=white)](https://ollama.ai)
[![DebianBadge](https://badges.debian.net/badges/debian/stable/docker/version.svg)](https://www.docker.com/)

This is the official implementation of the Open Source Institute-CoSMIC-CoachingWriter v1.0.0.


## 🌍 Overview

**CoSMIC Coaching Writer** is part of the **OpenSI Cognitive System of Machine Intelligent Computing (CoSMIC)** — a framework for explainable, responsible, and human-centered AI in education.

This service acts as an **academic writing coach**, not a ghostwriter.  
It analyzes drafts, guides learning, and teaches transferable writing principles using retrieval-augmented feedback and concept-based reasoning.

The system combines:
- **Retrieval-Augmented Generation (RAG)** for factual and contextual grounding.  
- **Ollama-based LLMs** (e.g., Qwen, Mistral) for efficient local inference.  
- **Ethical coaching behavior** enforced through structured prompts and filters.  
- **OpenAI / Ollama-compatible APIs** for seamless integration with existing tools.

---

## 🎓 Academic & Educational Focus

CoSMIC Coaching Writer is designed for **students, educators, and researchers** who want to:
- Receive constructive, structured feedback on their writing.  
- Learn *why* certain phrasing, structure, or tone matters.  
- Build academic writing competence through self-guided reflection.  

The coach never rewrites or generates essays.  
Instead, it helps users **analyze, reflect, and improve** their own work through guided conceptual feedback.

---

## ⚙️ Features

✅ **Retrieval-Augmented Context** — integrates sources from PDFs and text.  
✅ **Pedagogical Feedback Engine** — analyzes clarity, tone, argument, and structure.  
✅ **Model Flexibility** — supports interchangeable Ollama models via valves.  
✅ **Academic Integrity** — strictly prohibits rewriting or content generation.  
✅ **Multi-API Compatibility** — native support for both OpenAI and Ollama-style clients.  
✅ **Auto-Ingestion of Academic Documents** — syncs PDFs from OpenWebUI or `/uploads`.

---

## Architecture

```
                      +--------------------------------------+
                      |        OpenWebUI / Client UI         |
                      |  (Chat, Uploads, and Query Interface)|
                      +-------------------+------------------+
                                          |
                                          |  REST / HTTP (upload, query)
                                          v
    +---------------------------------------------------------------+
    |              🧩 CoSMIC Coaching Writer Service                 |
    |---------------------------------------------------------------|
    |                                                               |
    |   +----------------+         +-----------------------------+  |
    |   |   API Routes    | -----> |     Coaching Service        |  |
    |   | (FastAPI Layer) |         | (RAG + LLM Orchestration)  |  |
    |   +--------+--------+         +--------------+--------------+  |
    |            |                                 |                 |
    |            |                                 v                 |
    |            |                  +-----------------------------+  |
    |            |                  |   RAG Retriever             |  |
    |            |                  |  (Context + Source Lookup)  |  |
    |            |                  +---------------+-------------+  |
    |            |                                  |                |
    |            |                                  v                |
    |            |                  +-----------------------------+  |
    |            |                  |  Vector Database (FAISS)    |  |
    |            |                  |  + Embeddings (MiniLM)      |  |
    |            |                  +---------------+-------------+  |
    |            |                                  |                |
    |            |                                  v                |
    |            |                  +-----------------------------+  |
    |            +----------------> |   Ollama LLM Backend        |  |
    |                               |   (Qwen / Mistral models)   |  |
    |                               +-----------------------------+  |
    |                                                               |
    +---------------------------------------------------------------+
                                          |
                                          |  REST / WebSocket APIs
                                          v
                  +---------------------------------------------+
                  |        Pipelines Service (OpenWebUI)        |
                  |   - External pipeline integration           |
                  |   - Connects to Coaching Writer API         |
                  |   - Orchestrates academic workflows         |
                  +-------------------+-------------------------+
                                      |
                                      |  Feedback & Results
                                      v
                      +--------------------------------------+
                      |        OpenWebUI / Client UI         |
                      |   (Displays responses & insights)    |
                      +--------------------------------------+

```

### Components

| Layer | File | Role |
|-------|------|------|
| **Base Service** | `services/base.py` | Common service utilities (paths, logging). |
| **Retrieval** | `services/rag.py` | Retrieves and filters relevant text from the vector store. |
| **Generation** | `services/llm.py` | Wraps Ollama LLMs, enforces coaching rules, filters output. |
| **Orchestration** | `services/coach.py` | Combines RAG + LLM into context-aware coaching feedback. |
| **API Layer** | `routes/api_routes.py` | Exposes `/coach/query`, `/documents/*`, and OpenAI/Ollama endpoints. |

---

## 🌐 Endpoints

### 🔹 Primary REST API

| Method | Path | Description |
|--------|------|-------------|
| **GET** | `/health` | Returns service and active model status. |
| **POST** | `/documents/upload` | Upload a PDF (multipart/form-data). |
| **POST** | `/documents/ingest-text` | Ingest raw text into the vector database. |
| **GET** | `/documents` | List all ingested documents from the catalogue. |
| **POST** | `/coach/query` | Main endpoint — body `{query, use_rag, mode}` → structured coaching feedback. |

### 🔹 OpenAI-Compatible Endpoints

| Method | Path | Notes |
|--------|------|-------|
| **GET** | `/v1/models` | Lists the current backend model identifier. |
| **POST** | `/v1/chat/completions` | Accepts a standard `messages[]` array; prefix user message with `/norag` to disable retrieval. |

### 🔹 Ollama-Compatible Endpoints

Supported for UI and client interoperability:  
`/api/version`, `/api/tags`, `/api/show`, `/api/pull`, `/api/ps`, `/api/delete`, `/api/generate`, `/api/chat`.

---

### 🔸 Mode Parameter
The optional `mode` field customizes the coaching tone and focus:
- `critique` — detailed academic critique  
- `improve` — constructive clarity and coherence suggestions  
- `structure` — focus on argument and organization  
- `abstract` — research-summary guidance  
- `references` — citation and source-use advice  

---

## 🚀 Quick Start

This repository ships with a ready-to-run **Docker Compose** environment.  
It starts **Ollama** (model server), the **CoachingWriter API**, and **OpenWebUI** for interaction.

Detailed User Guide: https://docs.google.com/document/d/18pTrj2nETLMtE_Cn_deRucjL-H7XnJWmdP5Onv97Et0/edit?usp=sharing 

### 1️⃣ Start Services
```bash
docker compose up -d --build
```

2) Check all containers have completed startup and are running on Docker Desktop

3) Open the website

Open http://localhost:8080 in your browser.

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


