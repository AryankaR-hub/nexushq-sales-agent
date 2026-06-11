"""
API Routes — route handlers only, no business logic.
All heavy lifting is delegated to services.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.memory.factory import get_memory_backend
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteMemoryResponse,
    EvalAggregateResponse,
    HealthResponse,
    HistoryResponse,
    MessageSchema,
    SessionSchema,
)
from app.services.chat_service import ChatService
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

_CATALOG_PATH = Path(__file__).parents[2] / "catalog.json"


# ──────────────────────────────────────────────────────────────────────────────
# POST /chat/{user_id}
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/chat/{user_id}",
    response_model=ChatResponse,
    summary="Send a message and receive an AI response with eval",
)
async def chat(
    user_id: str,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Main chat endpoint.
    - Routes through the sales agent (tool use: search_catalog, get_user_memory)
    - Persists message + eval to DB
    - Extracts and stores new user facts for cross-session memory
    """
    try:
        service = ChatService(db)
        return await service.chat(
            user_id=user_id,
            user_message=body.message,
            session_id=body.session_id,
        )
    except Exception as e:
        logger.exception("Chat error for user %s: %s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ──────────────────────────────────────────────────────────────────────────────
# GET /chat/{user_id}/history
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/chat/{user_id}/history",
    response_model=HistoryResponse,
    summary="Full conversation history across all sessions",
)
async def get_history(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    memory = get_memory_backend(db)
    sessions = await memory.get_sessions_for_user(user_id)

    session_schemas = []
    for s in sessions:
        msgs = []
        for m in s.messages:
            eval_block = None
            if m.confidence is not None:
                eval_block = {
                    "groundedness": m.groundedness or 0.0,
                    "relevance": m.relevance or 0.0,
                    "confidence": m.confidence or 0.0,
                    "flagged": m.flagged or False,
                    "reasoning": m.eval_reasoning or "",
                }
            msgs.append(
                MessageSchema(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at,
                    eval=eval_block,
                    tools_called=m.tools_called,
                )
            )
        session_schemas.append(
            SessionSchema(
                session_id=s.id,
                created_at=s.created_at,
                updated_at=s.updated_at,
                messages=msgs,
            )
        )

    return HistoryResponse(
        user_id=user_id,
        total_sessions=len(session_schemas),
        sessions=session_schemas,
    )


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /chat/{user_id}/memory
# ──────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/chat/{user_id}/memory",
    response_model=DeleteMemoryResponse,
    summary="GDPR-style wipe of all user data",
)
async def delete_memory(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    memory = get_memory_backend(db)
    counts = await memory.delete_user_data(user_id)
    return DeleteMemoryResponse(
        user_id=user_id,
        sessions_deleted=counts["sessions_deleted"],
        memory_facts_deleted=counts["memory_facts_deleted"],
        message=f"All data for user '{user_id}' has been permanently deleted.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /catalog
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/catalog",
    summary="Returns the full product and pricing catalog",
)
async def get_catalog():
    with open(_CATALOG_PATH) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health(db: AsyncSession = Depends(get_db)):
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    return HealthResponse(
        status="ok",
        environment=settings.environment,
        db=db_status,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /chat/{user_id}/evals   (BONUS)
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/chat/{user_id}/evals",
    response_model=EvalAggregateResponse,
    summary="(Bonus) Aggregated eval stats across all sessions for a user",
)
async def get_evals(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    memory = get_memory_backend(db)
    stats = await memory.get_eval_stats(user_id)
    return EvalAggregateResponse(user_id=user_id, **stats)
