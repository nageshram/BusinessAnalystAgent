"""
API routes — FastAPI endpoints for the Business Analyst Agent.

Endpoints:
- POST /api/chat          — Send a message (streaming SSE response)
- POST /api/chat/sync     — Send a message (non-streaming, full response)
- GET  /api/conversations  — List conversations (paginated)
- GET  /api/conversations/{thread_id}/messages — Get message history
- POST /api/upload/dataset  — Upload CSV/XLSX dataset for analysis
- POST /api/upload/document — Upload document for knowledge base (RAG)
- GET  /api/datasets        — List uploaded datasets
- GET  /api/datasets/{dataset_id} — Dataset details + profile
- GET  /health             — Health check
- GET  /health/detailed    — Detailed health check with component status
"""

import uuid
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.db.database import check_db_health
from app.db.crud import (
    get_or_create_conversation,
    list_conversations,
    get_chat_history,
    save_chat_message,
    save_dataset,
    get_dataset,
    list_datasets,
)
from app.agent.graph import get_agent
from app.data.manager import DatasetManager
from app.rag.ingestion import add_document_to_rag
from app.rag.retriever import get_document_stats
from app.circuit_breaker import llm_circuit_breaker, db_circuit_breaker, rag_circuit_breaker

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

Path("uploads").mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    thread_id: Optional[str] = Field(None, description="Thread ID (auto-generated if not provided)")
    model_name: Optional[str] = Field(None, description="LLM model to use")
    dataset_id: Optional[str] = Field(None, description="Active dataset ID for analysis context")


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    model_name: str


# ──────────────────────────────────────────────
# Chat Endpoints
# ──────────────────────────────────────────────

@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a message and receive a streaming SSE response.
    Token-by-token streaming for a ChatGPT-like experience.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    model_name = request.model_name or settings.OPENROUTER_MODEL
    dataset_id = request.dataset_id or ""

    # Ensure conversation exists
    get_or_create_conversation(
        thread_id=thread_id,
        first_message=request.message,
        model_name=model_name,
    )

    # Save the user's message
    save_chat_message(thread_id=thread_id, role="human", content=request.message)

    # Get agent
    agent = get_agent(model_name)
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        """Generate SSE events from the agent's streaming response."""
        full_response = []

        try:
            for message_chunk, metadata in agent.stream(
                {
                    "messages": [HumanMessage(content=request.message)],
                    "step_count": 0,
                    "thread_id": thread_id,
                    "model_name": model_name,
                    "active_dataset_id": dataset_id,
                },
                config=config,
                stream_mode="messages",
            ):
                if message_chunk.content:
                    full_response.append(message_chunk.content)
                    data = json.dumps({"content": message_chunk.content, "thread_id": thread_id})
                    yield f"data: {data}\n\n"

            # Save the full AI response
            complete_response = "".join(full_response)
            if complete_response:
                save_chat_message(
                    thread_id=thread_id,
                    role="ai",
                    content=complete_response,
                )

            yield f"data: {json.dumps({'done': True, 'thread_id': thread_id})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_data = json.dumps({"error": str(e)[:200], "thread_id": thread_id})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Thread-ID": thread_id,
        },
    )


@router.post("/api/chat", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """
    Send a message and receive the complete response (non-streaming).
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    model_name = request.model_name or settings.OPENROUTER_MODEL
    dataset_id = request.dataset_id or ""

    # Ensure conversation exists
    get_or_create_conversation(
        thread_id=thread_id,
        first_message=request.message,
        model_name=model_name,
    )

    # Save user message
    save_chat_message(thread_id=thread_id, role="human", content=request.message)

    try:
        agent = get_agent(model_name)
        config = {"configurable": {"thread_id": thread_id}}

        result = agent.invoke(
            {
                "messages": [HumanMessage(content=request.message)],
                "step_count": 0,
                "thread_id": thread_id,
                "model_name": model_name,
                "active_dataset_id": dataset_id,
            },
            config=config,
        )

        # Extract the last AI message
        ai_messages = [
            m for m in result.get("messages", [])
            if hasattr(m, 'type') and m.type == 'ai' and m.content
        ]

        response_text = ai_messages[-1].content if ai_messages else "No response generated."

        # Save AI response
        save_chat_message(thread_id=thread_id, role="ai", content=response_text)

        return ChatResponse(
            thread_id=thread_id,
            response=response_text,
            model_name=model_name,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)[:200]}")


# ──────────────────────────────────────────────
# Conversation Endpoints
# ──────────────────────────────────────────────

@router.get("/api/conversations")
async def api_list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    include_archived: bool = Query(False),
):
    """List conversations with pagination."""
    result = list_conversations(page=page, page_size=page_size, include_archived=include_archived)

    conversations = [
        {
            "thread_id": c.thread_id,
            "title": c.title,
            "model_name": c.model_name,
            "message_count": c.message_count,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in result["conversations"]
    ]

    return {
        "conversations": conversations,
        "total": result["total"],
        "page": result["page"],
        "page_size": result["page_size"],
        "has_next": result["has_next"],
    }


@router.get("/api/conversations/{thread_id}/messages")
async def api_get_messages(
    thread_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Get chat message history for a conversation."""
    messages = get_chat_history(thread_id=thread_id, limit=limit, offset=offset)

    return {
        "thread_id": thread_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
        "count": len(messages),
    }


# ──────────────────────────────────────────────
# Dataset Upload Endpoints
# ──────────────────────────────────────────────

@router.post("/api/upload/dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    thread_id: str = Query(..., description="Thread ID to associate the dataset with"),
):
    """
    Upload a CSV/XLSX/JSON dataset for analysis.
    Max size: 50MB. Max rows: 50,000.
    """
    # Read file content
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    # Save to disk temporarily
    filename = file.filename or "uploaded_dataset"
    temp_path = Path("uploads") / f"temp_{filename}"
    temp_path.write_bytes(contents)

    try:
        # Process with DatasetManager
        result = DatasetManager.upload_dataset(
            file_path=str(temp_path),
            original_filename=filename,
            thread_id=thread_id,
        )

        # Save to database
        save_dataset(
            dataset_id=result["dataset_id"],
            name=result["name"],
            file_path=result["file_path"],
            file_type=result["file_type"],
            row_count=result["row_count"],
            column_count=result["column_count"],
            file_size_bytes=result.get("file_size_bytes"),
            thread_id=thread_id,
        )

        # Clean up temp file
        temp_path.unlink(missing_ok=True)

        return result

    except ValueError as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        logger.error(f"Dataset upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)[:200]}")


@router.post("/api/upload/document")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Query(..., description="Thread ID to associate the document with"),
):
    """
    Upload a document (PDF, DOCX, TXT, MD) for the RAG knowledge base.
    """
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    filename = file.filename or "uploaded_document"
    file_path = Path("uploads") / f"{thread_id}_{filename}"
    file_path.write_bytes(contents)

    try:
        result = add_document_to_rag(
            file_path=str(file_path),
            thread_id=thread_id,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Document upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)[:200]}")


# ──────────────────────────────────────────────
# Dataset Endpoints
# ──────────────────────────────────────────────

@router.get("/api/datasets")
async def api_list_datasets(
    thread_id: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List uploaded datasets with optional thread_id filter."""
    result = list_datasets(thread_id=thread_id, page=page, page_size=page_size)

    datasets = [
        {
            "dataset_id": d.dataset_id,
            "name": d.name,
            "file_type": d.file_type,
            "row_count": d.row_count,
            "column_count": d.column_count,
            "status": d.status,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in result["datasets"]
    ]

    return {
        "datasets": datasets,
        "total": result["total"],
        "page": result["page"],
        "page_size": result["page_size"],
        "has_next": result["has_next"],
    }


@router.get("/api/datasets/{dataset_id}")
async def api_get_dataset(dataset_id: str):
    """Get dataset details and profile."""
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    result = {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "file_type": dataset.file_type,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "status": dataset.status,
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
    }

    if dataset.profile_json:
        result["profile"] = json.loads(dataset.profile_json)

    return result


@router.get("/api/documents/{thread_id}/stats")
async def api_document_stats(thread_id: str):
    """Get statistics about uploaded knowledge base documents."""
    return get_document_stats(thread_id)


# ──────────────────────────────────────────────
# Health Check Endpoints
# ──────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Simple health check — returns 200 if the service is running."""
    return {"status": "healthy", "version": settings.APP_VERSION}


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    db_health = check_db_health()

    circuit_breakers = {
        "llm": llm_circuit_breaker.get_status(),
        "database": db_circuit_breaker.get_status(),
        "rag": rag_circuit_breaker.get_status(),
    }

    overall = "healthy" if db_health["status"] == "healthy" else "degraded"
    for name, cb in circuit_breakers.items():
        if cb["state"] == "open":
            overall = "degraded"
            break

    return {
        "status": overall,
        "version": settings.APP_VERSION,
        "components": {
            "database": db_health,
            "circuit_breakers": circuit_breakers,
        },
    }
