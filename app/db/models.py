"""
ORM Models
──────────
Session       — one logical conversation window (uuid, user_id, timestamps)
Message       — individual turns within a session (role, content, eval, tools)
UserMemory    — persistent cross-session facts extracted per user
FlaggedEvent  — entries created when flag_for_human is invoked
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    String, Text, Float, Boolean, DateTime, ForeignKey, Integer, JSON
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan",
        order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)   # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Eval block — only populated on assistant messages
    groundedness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    relevance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    flagged: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    eval_reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tools_called: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    session: Mapped["Session"] = relationship("Session", back_populates="messages")


class UserMemory(Base):
    """
    Stores extracted facts about a user across sessions.
    Each row is one atomic fact (e.g. "interested in Enterprise SSO").
    When compression kicks in, older messages are summarised into a
    single row with fact_type='summary'.
    """
    __tablename__ = "user_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    fact_type: Mapped[str] = mapped_column(String(50), nullable=False, default="fact")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class FlaggedEvent(Base):
    __tablename__ = "flagged_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
