"""
api_routes.py
-------------

This module defines the REST API endpoints for the **CoSMIC Coaching Writer** service.  
It provides three categories of interfaces:

1. **Custom Coaching API**  
   - `/coach/query`: Core endpoint for handling coaching queries.  
   - `/documents/*`: Uploading, ingesting, and listing documents.  

2. **OpenAI-Compatible API**  
   Minimal endpoints (`/v1/models`, `/v1/chat/completions`) to allow clients 
   expecting an OpenAI API surface to interact seamlessly with the Coaching Writer.  

3. **Ollama-Compatible API**  
   Minimal endpoints (`/api/*`) for interoperability with Ollama-style clients.  

Key Features:
- Retrieval-Augmented Generation (RAG) toggle support (`/norag` directive).  
- Supports streaming and non-streaming responses.  
- Provides health check endpoint for monitoring.  
- Normalizes incoming model names to the configured base LLM.  
"""

from fastapi import APIRouter, UploadFile, Query, File, HTTPException
from fastapi.responses import StreamingResponse
import os, shutil, time, uuid, json, glob, csv
from typing import List, Union, Optional
from ..schemas.requests import QueryRequest, TextIngestRequest
from ..schemas.responses import HealthResponse, QueryResponse
from ..services.coach import coach_singleton
from ..services.vector_database import vector_db_singleton
from ..core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", llm=settings.llm_name)


DEFAULT_WEBUI_DATA = "/app/backend/data"
DEFAULT_UPLOADS_DIR = os.path.join(DEFAULT_WEBUI_DATA, "uploads")


def _resolve_attachment_path(item: Union[str, dict]) -> Optional[str]:
    """Best-effort resolve an OpenWebUI attachment record to a file path.

    Supports strings (absolute paths) or dicts with keys like 'path', 'filepath', 'file_path',
    or just a 'name' which we search under the shared OpenWebUI data folder.
    """
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
            if isinstance(path, str) and path.startswith("file://"):
                path = path[len("file://"):]
            # If path looks relative to backend data, join it
            if isinstance(path, str) and not os.path.isabs(path):
                cand = os.path.join(base, path.lstrip("/"))
                return cand
            return path
        name = item.get("name") or item.get("filename")
        if name:
            # Common upload folder in OpenWebUI
            cand1 = os.path.join(base, "uploads", name)
            if os.path.exists(cand1):
                return cand1
            cand2 = os.path.join(base, name)
            if os.path.exists(cand2):
                return cand2
    return None


@router.post("/coach/query", response_model=QueryResponse)
def coach_query(payload: QueryRequest):
    # Ingest any referenced PDF attachments first, if accessible via the shared volume
    docs = payload.documents or []
    resolved_count = 0
    attached_context_parts: list[str] = []
    for d in docs:
        try:
            path = _resolve_attachment_path(d)
            if path and path.lower().endswith(".pdf") and os.path.exists(path):
                # Extract a small snippet to inject immediate context
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(path)
                    pages = loader.load()[:2]  # first couple of pages as context
                    snippet = "\n".join([p.page_content for p in pages])
                    if snippet:
                        attached_context_parts.append(snippet[:3000])
                except Exception:
                    pass
                # Also add to vector db for future queries
                vector_db_singleton.add_pdf(path)
                resolved_count += 1
        except Exception:
            # Swallow per-file issues to keep query flowing
            pass

    # Fallback: if no explicit attachments were provided or resolved, try to ingest
    # any new PDFs from the default OpenWebUI uploads folder that are not in catalogue yet.
    if resolved_count == 0 and os.path.isdir(DEFAULT_UPLOADS_DIR):
        try:
            ingested = _sync_new_pdfs_from_uploads()
            if ingested:
                print(f"[coach_query] Auto-synced {ingested} PDF(s) from uploads")
        except Exception as e:
            print(f"[coach_query] Auto-sync error: {e}")
        # Also try injecting context from the most recent uploaded PDF
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
    result = coach_singleton(payload.query, use_rag=payload.use_rag, mode=payload.mode, injected_context=injected_context)
    return QueryResponse(**result)


@router.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    temp_path = os.path.join("uploads", file.filename)
    with open(temp_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    if file.filename.lower().endswith('.pdf'):
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
                    parts = line.split(',')
                    if len(parts) >= 2:
                        catalogue.append({"source": parts[0], "time": parts[1]})
    return {"documents": catalogue}


def _catalogue_sources_set() -> set:
    """Read the catalogue CSV and return a set of recorded sources."""
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
    """Ingest PDFs from OpenWebUI uploads directory not yet present in catalogue.

    Args:
        limit: Max number of PDFs to consider (most recent first).

    Returns:
        int: number of PDFs ingested.
    """
    if not os.path.isdir(DEFAULT_UPLOADS_DIR):
        return 0
    recorded = _catalogue_sources_set()
    pdfs = glob.glob(os.path.join(DEFAULT_UPLOADS_DIR, "**", "*.pdf"), recursive=True)
    # sort by mtime desc
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


@router.post("/documents/sync")
def documents_sync():
    """Manually sync new PDFs from OpenWebUI uploads directory, skipping existing ones."""
    ingested = _sync_new_pdfs_from_uploads(limit=100)
    return {"status": "ok", "ingested": ingested}
def _base_model_id() -> str:
    return settings.llm_name.replace("ollama:", "")


def _normalize_incoming_model(model: str) -> str:
    if not model:
        return _base_model_id()
    lower = model.strip().lower()
    if lower in {"gemma2:2b", "gemma2", "gemma:2b"}:
        return _base_model_id()
    return _base_model_id()


@router.get("/v1/models")
def openai_models():
    base_llm = settings.llm_name.replace("ollama:", "")
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
    system_context_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="at least one user message required")
    query = user_messages[-1]
    trimmed = query.strip()
    use_rag = True
    if trimmed.lower().startswith('/norag'):
        use_rag = False
        query = trimmed[len('/norag'):].strip(': ').strip()

    result = coach_singleton(query, use_rag=use_rag, mode=None)
    answer = result.get("response", "")
    sources = result.get("sources", [])

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    requested_model = body.get("model", settings.llm_name.replace("ollama:", ""))
    model_name = _normalize_incoming_model(requested_model)
    base_llm = settings.llm_name.replace("ollama:", "")
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
            "underlying_model": base_llm,
            "sources": sources,
        },
    }


@router.get("/api/version")
def ollama_version():
    return {"version": "v0.1.0-cosmic"}


@router.get("/api/tags")
def ollama_tags():
    from datetime import datetime
    base_llm = settings.llm_name.replace("ollama:", "")
    now = datetime.utcnow().isoformat() + "Z"
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
    base_llm = settings.llm_name.replace("ollama:", "")
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


@router.post("/api/generate")
def ollama_generate(body: dict):
    model = _normalize_incoming_model(body.get("model"))
    prompt = (body.get("prompt") or "").strip()
    stream = bool(body.get("stream", True))
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")
    result = coach_singleton(prompt, use_rag=True, mode=None)
    answer = result.get("response", "")
    if not stream:
        return {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "response": answer,
            "done": True,
        }
    def _iter():
        words = answer.split()
        for w in words:
            yield ("{" f"\"model\":\"{model}\",\"created_at\":\"" + time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()) + "\"," "\"response\":" + json.dumps(w + " ") + ",\"done\":false}\n")
        yield ("{\"model\":\"" + model + "\",\"created_at\":\"" + time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()) + "\",\"response\":\"\",\"done\":true}\n")
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
    answer = result.get("response", "")
    if not stream:
        return {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "message": {"role": "assistant", "content": answer},
            "done": True,
        }
    def _iter():
        parts = answer.split()
        for p in parts:
            yield ("{" + f"\"model\":\"{model}\",\"created_at\":\"" + time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()) + "\"," + "\"message\":" + json.dumps({"role": "assistant", "content": p + " "}) + ",\"done\":false}\n")
        yield ("{" + f"\"model\":\"{model}\",\"created_at\":\"" + time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()) + "\"," + "\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done_reason\":\"stop\",\"done\":true}\n")
    return StreamingResponse(_iter(), media_type="application/x-ndjson")

    def _iter():
        # naive chunking by words
        words = answer.split()
        for w in words:
            yield (\
                "{"\
                f"\"model\":\"{model}\",\"created_at\":\"" + time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()) + "\","\
                "\"response\":" + json.dumps(w + " ") + ",\"done\":false}"\
                + "\n"\
            )
        yield ("{\"model\":\"" + model + "\",\"created_at\":\"" + time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()) + "\",\"response\":\"\",\"done\":true}\n")

    return StreamingResponse(_iter(), media_type="application/x-ndjson")