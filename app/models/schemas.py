"""
Pydantic schemas for request bodies and API responses.
Kept strictly in this module — no business logic here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Request bodies
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User's message")
    session_id: Optional[str] = Field(
        None,
        description="Optional: resume a specific session. If omitted, a new session is created.",
    )


# ──────────────────────────────────────────────
# Eval block
# ──────────────────────────────────────────────

class EvalBlock(BaseModel):
    groundedness: float = Field(..., ge=0.0, le=1.0)
    relevance: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    flagged: bool
    reasoning: str


# ──────────────────────────────────────────────
# Chat responses
# ──────────────────────────────────────────────

class ChatResponse(BaseModel):
    response: str
    eval: EvalBlock
    tools_called: list[str]
    session_id: str
    user_id: str


# ──────────────────────────────────────────────
# History
# ──────────────────────────────────────────────

class MessageSchema(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    eval: Optional[EvalBlock] = None
    tools_called: Optional[list[str]] = None

    class Config:
        from_attributes = True


class SessionSchema(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageSchema]

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    user_id: str
    total_sessions: int
    sessions: list[SessionSchema]


# ──────────────────────────────────────────────
# Evals aggregation (bonus endpoint)
# ──────────────────────────────────────────────

class EvalAggregateResponse(BaseModel):
    user_id: str
    total_responses: int
    avg_groundedness: float
    avg_relevance: float
    avg_confidence: float
    flagged_count: int
    high_confidence_pct: float   # % responses with confidence >= 0.80


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    environment: str
    db: str


# ──────────────────────────────────────────────
# Memory wipe
# ──────────────────────────────────────────────

class DeleteMemoryResponse(BaseModel):
    user_id: str
    sessions_deleted: int
    memory_facts_deleted: int
    message: str
