"""
api_routes.py
-------------

Implements all REST and API-compatible endpoints for the CoSMIC Coaching Writer.

### Overview
This router provides three integration layers:

1. **Custom Coaching API**
   - `/coach/query` — main interface for text coaching.
   - `/documents/*` — ingest, upload, and synchronization endpoints.

2. **OpenAI-Compatible API**
   - `/v1/models`
   - `/v1/chat/completions`
   Enables use through OpenAI-compatible clients.

3. **Ollama-Compatible API**
   - `/api/*` endpoints to enable use with Ollama tooling or OpenWebUI.

### Features
- Automatic ingestion of new PDFs from OpenWebUI uploads.
- Retrieval-Augmented Generation toggle (`/norag` command prefix).
- Supports both standard and streaming responses.
- Auto-initializes vector database catalogue from existing uploads.
"""

from fastapi import APIRouter, UploadFile, Query, File, HTTPException
from fastapi.responses import StreamingResponse
import os, shutil, time, uuid, json, glob, csv, types
from typing import List, Union, Optional
from ..schemas.requests import QueryRequest, TextIngestRequest
from ..schemas.responses import HealthResponse, QueryResponse
from ..services.coach import coach_singleton
from ..services.vector_database import vector_db_singleton
from ..core.config import settings
import warnings
warnings.filterwarnings("ignore", message="Ignoring wrong pointing object")

router = APIRouter()

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", llm=settings.llm_name)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

DEFAULT_WEBUI_DATA = "/app/backend/data"
DEFAULT_UPLOADS_DIR = os.path.join(DEFAULT_WEBUI_DATA, "uploads")


def _resolve_attachment_path(item: Union[str, dict]) -> Optional[str]:
    """Resolve OpenWebUI-style attachment metadata into a local file path."""
    base = DEFAULT_WEBUI_DATA
    if isinstance(item, str):
        if item.startswith("file://"):
            item = item[len("file://"):]
        return item

    if isinstance(item, dict):
        path = (
            item.get("path")
            or item.get("filepath")
            or item.get("file_path")
            or item.get("diskPath")
            or None
        )
        if path:
            if path.startswith("file://"):
                path = path[len("file://"):]
            if not os.path.isabs(path):
                return os.path.join(base, path.lstrip("/"))
            return path

        name = item.get("name") or item.get("filename")
        if name:
            cand1 = os.path.join(base, "uploads", name)
            cand2 = os.path.join(base, name)
            for c in (cand1, cand2):
                if os.path.exists(c):
                    return c
    return None


def _catalogue_sources_set() -> set:
    """Return set of recorded document sources from catalogue CSV."""
    recorded = set()
    if os.path.exists(vector_db_singleton.catalogue_path):
        try:
            with open(vector_db_singleton.catalogue_path, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if row:
                        recorded.add(row[0])
        except Exception:
            pass
    return recorded


def _sync_new_pdfs_from_uploads(limit: int = 20) -> int:
    """Ingest new PDFs from OpenWebUI uploads directory not in the catalogue."""
    if not os.path.isdir(DEFAULT_UPLOADS_DIR):
        return 0

    recorded = _catalogue_sources_set()
    pdfs = glob.glob(os.path.join(DEFAULT_UPLOADS_DIR, "**", "*.pdf"), recursive=True)
    pdfs.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    count = 0
    for p in pdfs[:limit]:
        if p not in recorded and os.path.exists(p):
            try:
                vector_db_singleton.add_pdf(p)
                count += 1
            except Exception:
                pass
    return count


def _base_model_id() -> str:
    """Return the active LLM name minus the ollama: prefix."""
    return settings.llm_name.replace("ollama:", "")


def _normalize_incoming_model(model: Optional[str]) -> str:
    """Normalize any incoming model name to match the active LLM."""
    return _base_model_id()


# ---------------------------------------------------------------------------
# Custom Coaching API
# ---------------------------------------------------------------------------

@router.post("/coach/query", response_model=QueryResponse)
def coach_query(payload: QueryRequest):
    """Handle incoming coaching requests and optional document attachments."""
    docs = payload.documents or []
    resolved_count = 0
    attached_context_parts: list[str] = []

    for d in docs:
        try:
            path = _resolve_attachment_path(d)
            if path and path.lower().endswith(".pdf") and os.path.exists(path):
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(path)
                pages = loader.load()[:2]
                snippet = "\n".join([p.page_content for p in pages])
                if snippet:
                    attached_context_parts.append(snippet[:3000])
                vector_db_singleton.add_pdf(path)
                resolved_count += 1
        except Exception:
            pass

    if resolved_count == 0 and os.path.isdir(DEFAULT_UPLOADS_DIR):
        try:
            ingested = _sync_new_pdfs_from_uploads()
            if ingested:
                print(f"[coach_query] Auto-synced {ingested} PDF(s) from uploads")
        except Exception as e:
            print(f"[coach_query] Auto-sync error: {e}")

        try:
            pdfs = glob.glob(os.path.join(DEFAULT_UPLOADS_DIR, "**", "*.pdf"), recursive=True)
            if pdfs:
                pdfs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                latest = pdfs[0]
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(latest)
                pages = loader.load()[:2]
                snippet = "\n".join([p.page_content for p in pages])
                if snippet:
                    attached_context_parts.append(snippet[:3000])
        except Exception:
            pass

    injected_context = "\n---\n".join(attached_context_parts) if attached_context_parts else None
    result = coach_singleton(
        payload.query,
        use_rag=payload.use_rag,
        mode=payload.mode,
        injected_context=injected_context,
    )
    return QueryResponse(**result)


@router.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    temp_path = os.path.join("uploads", file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if file.filename.lower().endswith(".pdf"):
        vector_db_singleton.add_pdf(temp_path)
    return {"status": "uploaded", "filename": file.filename}


@router.post("/documents/ingest-text")
def ingest_text(payload: TextIngestRequest):
    status = vector_db_singleton.add_text(payload.text)
    return {"status": "added" if status == 0 else "skipped"}


@router.post("/documents/ingest-folder")
def ingest_folder(path: str = Query(..., description="Path to folder containing PDFs")):
    count = vector_db_singleton.add_pdf_folder(path)
    return {"status": "completed", "pdfs_added": count}


@router.get("/documents")
def list_documents():
    catalogue = []
    if os.path.exists(vector_db_singleton.catalogue_path):
        with open(vector_db_singleton.catalogue_path) as f:
            lines = f.read().strip().splitlines()
            if lines:
                for line in lines[1:]:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        catalogue.append({"source": parts[0], "time": parts[1]})
    return {"documents": catalogue}

@router.post("/documents/sync")
def documents_sync():
    ingested = _sync_new_pdfs_from_uploads(limit=100)
    return {"status": "ok", "ingested": ingested}


# ---------------------------------------------------------------------------
# OpenAI-Compatible Endpoints
# ---------------------------------------------------------------------------

@router.get("/v1/models")
def openai_models():
    base_llm = _base_model_id()
    return {
        "object": "list",
        "data": [
            {
                "id": base_llm,
                "object": "model",
                "owned_by": "coaching-writer",
                "name": base_llm,
                "description": f"{base_llm} via CoachingWriter",
            }
        ],
    }


@router.post("/v1/chat/completions")
def openai_chat_completions(body: dict):
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages field required")

    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="at least one user message required")

    query = user_messages[-1].strip()
    use_rag = not query.lower().startswith("/norag")
    if not use_rag:
        query = query[len("/norag"):].strip(": ").strip()

    result = coach_singleton(query, use_rag=use_rag, mode=None)
    answer = result.get("response", "")
    sources = result.get("sources", [])

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = _normalize_incoming_model(body.get("model"))

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "cosmic": {
            "rag_used": use_rag,
            "source_count": len(sources),
            "underlying_model": _base_model_id(),
            "sources": sources,
        },
    }


# ---------------------------------------------------------------------------
# Ollama-Compatible Endpoints
# ---------------------------------------------------------------------------

@router.get("/api/version")
def ollama_version():
    return {"version": "v0.1.0-cosmic"}


@router.get("/api/tags")
def ollama_tags():
    from datetime import datetime
    now = datetime.utcnow().isoformat() + "Z"
    base_llm = _base_model_id()
    return {
        "models": [
            {
                "name": base_llm,
                "model": base_llm,
                "modified_at": now,
                "size": 0,
                "digest": "",
                "details": {"format": "gguf", "family": "ollama", "families": ["ollama"]},
            }
        ]
    }


@router.get("/api/show")
def ollama_show(name: str):
    from datetime import datetime
    now = datetime.utcnow().isoformat() + "Z"
    base_llm = _base_model_id()
    return {
        "model": {
            "name": base_llm,
            "model": base_llm,
            "modified_at": now,
            "size": 0,
            "digest": "",
            "details": {"format": "gguf", "family": "ollama", "families": ["ollama"]},
        }
    }


@router.post("/api/pull")
def ollama_pull(body: dict):
    name = (body or {}).get("name", "")
    return {"status": "success", "name": name}


@router.get("/api/ps")
def ollama_ps():
    return {"models": []}


@router.delete("/api/delete")
def ollama_delete(body: dict):
    return {"status": "success"}

def _current_ts():
    """Return the current UTC timestamp in Ollama-style format."""
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

@router.post("/api/generate")
def ollama_generate(body: dict):
    model = _normalize_incoming_model(body.get("model"))
    prompt = (body.get("prompt") or "").strip()
    stream = bool(body.get("stream", True))
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")

    result = coach_singleton(prompt, use_rag=True, mode=None)

    # Normalize all possible return types to a plain text answer generator
    def _normalize_result(res):
        # case 1: direct dict
        if isinstance(res, dict):
            yield res.get("response", "")
        # case 2: list of chunks
        elif isinstance(res, list):
            for chunk in res:
                if isinstance(chunk, dict) and chunk.get("response"):
                    yield chunk["response"]
        # case 3: generator
        elif isinstance(res, types.GeneratorType):
            for chunk in res:
                if isinstance(chunk, dict) and chunk.get("response"):
                    yield chunk["response"]
        # case 4: plain string
        elif isinstance(res, str):
            yield res
        else:
            yield str(res)

    if not stream:
        full_text = "".join(_normalize_result(result))
        return {
            "model": model,
            "created_at": _current_ts(),
            "response": full_text,
            "done": True,
        }

    def _iter():
        for token in _normalize_result(result):
            # Filter out any 'thinking' text if model emitted it
            if "thinking" in token.lower():
                continue
            yield json.dumps({
                "model": model,
                "created_at": _current_ts(),
                "response": token,
                "done": False
            }) + "\n"
        yield json.dumps({
            "model": model,
            "created_at": _current_ts(),
            "response": "",
            "done": True
        }) + "\n"

    return StreamingResponse(_iter(), media_type="application/x-ndjson")


@router.post("/api/chat")
def ollama_chat(body: dict):
    model = _normalize_incoming_model(body.get("model"))
    messages = body.get("messages") or []
    stream = bool(body.get("stream", True))
    if not messages:
        raise HTTPException(status_code=400, detail="messages required")

    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="user message required")

    query = user_msgs[-1]
    result = coach_singleton(query, use_rag=True, mode=None)

    def _normalize_result(res):
        if isinstance(res, dict):
            yield res.get("response", "")
        elif isinstance(res, list):
            for chunk in res:
                if isinstance(chunk, dict) and chunk.get("response"):
                    yield chunk["response"]
        elif isinstance(res, types.GeneratorType):
            for chunk in res:
                if isinstance(chunk, dict) and chunk.get("response"):
                    yield chunk["response"]
        elif isinstance(res, str):
            yield res
        else:
            yield str(res)

    if not stream:
        full_text = "".join(_normalize_result(result))
        return {
            "model": model,
            "created_at": _current_ts(),
            "message": {"role": "assistant", "content": full_text},
            "done": True,
        }

    def _iter():
        for token in _normalize_result(result):
            if "thinking" in token.lower():
                continue
            yield json.dumps({
                "model": model,
                "created_at": _current_ts(),
                "message": {"role": "assistant", "content": token},
                "done": False
            }) + "\n"
        yield json.dumps({
            "model": model,
            "created_at": _current_ts(),
            "message": {"role": "assistant", "content": ""},
            "done_reason": "stop",
            "done": True
        }) + "\n"

    return StreamingResponse(_iter(), media_type="application/x-ndjson")