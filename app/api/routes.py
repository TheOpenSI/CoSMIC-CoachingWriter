from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import os, shutil, time, uuid, json
from ..schemas.requests import QueryRequest, TextIngestRequest
from ..schemas.responses import HealthResponse, QueryResponse
from ..services.coach import coach_singleton
from ..services.vector_database import vector_db_singleton
from ..core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", llm=settings.llm_name)


@router.post("/coach/query", response_model=QueryResponse)
def coach_query(payload: QueryRequest):
    result = coach_singleton(payload.query, use_rag=payload.use_rag, mode=payload.mode)
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
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse
import os, shutil, time, uuid, json
from ..schemas.requests import QueryRequest, TextIngestRequest
from ..schemas.responses import HealthResponse, QueryResponse
from ..services.coach import coach_singleton
from ..services.vector_database import vector_db_singleton
from ..core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", llm=settings.llm_name)


@router.post("/coach/query", response_model=QueryResponse)
def coach_query(payload: QueryRequest):
    result = coach_singleton(payload.query, use_rag=payload.use_rag, mode=payload.mode)
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

# ---------------- OpenAI-Compatible Minimal Endpoints -----------------
# This allows pointing OPENAI_API_BASE_URL directly to this service,
# avoiding the need for the separate pipelines container if desired.

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
    # Expect: {model: str, messages: [{role, content}], stream?: bool, temperature?: float, max_tokens?: int}
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages field required")
    # Aggregate system prompts as context (joined); last user content is the query.
    system_context_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="at least one user message required")
    query = user_messages[-1]
    context = "\n".join(system_context_parts) if system_context_parts else ""

    # Basic directive: if user starts with /norag disable retrieval
    use_rag = True
    trimmed = query.strip()
    if trimmed.lower().startswith('/norag'):
        use_rag = False
        query = trimmed[len('/norag'):].strip(': ').strip()

    result = coach_singleton(query, use_rag=use_rag, mode=None)
    answer = result.get("response", "")
    sources = result.get("sources", [])
    if sources and use_rag:
        # Optionally append brief first source snippet if not already present
        pass

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    requested_model = body.get("model", settings.llm_name.replace("ollama:", ""))
    # Normalize only to the base model name used by the backend
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


# ---------------- Ollama-Compatible Minimal Endpoints -----------------

def _base_model_id() -> str:
    return settings.llm_name.replace("ollama:", "")


def _normalize_incoming_model(model: str) -> str:
    if not model:
        return _base_model_id()
    lower = model.strip().lower()
    # Map common variants to the configured base model id
    if lower in {"gemma2:2b", "gemma2", "gemma:2b"}:
        return _base_model_id()
    return _base_model_id()


@router.get("/api/version")
def ollama_version():
    # Minimal version info
    return {"version": "v0.1.0-cosmic"}


@router.get("/api/tags")
def ollama_tags():
    # Expose only the underlying base model so admin screens can pick it;
    # avoid listing the alias here to prevent duplicates in UI.
    from datetime import datetime

    base_llm = settings.llm_name.replace("ollama:", "")
    now = datetime.utcnow().isoformat() + "Z"
    return {
        "models": [
            {
                "name": base_llm,   # e.g., gemma2:2b
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
    # Provide model details; accept alias or base model names
    from datetime import datetime
    now = datetime.utcnow().isoformat() + "Z"
    base_llm = settings.llm_name.replace("ollama:", "")
    name_lower = (name or "").strip().lower()
    # Default to base model
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
    # No-op pull that always succeeds to keep UI happy
    name = (body or {}).get("name", "")
    # Optionally stream, but we just return a success
    return {"status": "success", "name": name}


@router.get("/api/ps")
def ollama_ps():
    # Return empty running models list
    return {"models": []}


@router.delete("/api/delete")
def ollama_delete(body: dict):
    # No-op delete, always succeed
    return {"status": "success"}


@router.post("/api/generate")
def ollama_generate(body: dict):
    # Expect: {model: str, prompt: str, stream?: bool}
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


@router.post("/api/chat")
def ollama_chat(body: dict):
    # Expect: {model: str, messages: [...], stream?: bool}
    model = _normalize_incoming_model(body.get("model"))
    messages = body.get("messages") or []
    stream = bool(body.get("stream", True))

    if not messages:
        raise HTTPException(status_code=400, detail="messages required")

    # Aggregate system and use the last user message
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
        # stream in small chunks
        parts = answer.split()
        for p in parts:
            yield (
                "{" +
                f"\"model\":\"{model}\",\"created_at\":\"" + time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()) + "\"," +
                "\"message\":" + json.dumps({"role": "assistant", "content": p + " "}) + ",\"done\":false}\n"
            )
        yield (
            "{" +
            f"\"model\":\"{model}\",\"created_at\":\"" + time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()) + "\"," +
            "\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done_reason\":\"stop\",\"done\":true}\n"
        )

    return StreamingResponse(_iter(), media_type="application/x-ndjson")
