"""
SQLite / Postgres backend (SQLAlchemy async).
Swapping to Postgres = change DATABASE_URL env var only.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Session as DBSession, Message as DBMessage, UserMemory
from app.memory.base import AbstractMemoryBackend, MemoryFact, StoredMessage, StoredSession


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_stored_message(m: DBMessage) -> StoredMessage:
    return StoredMessage(
        id=m.id,
        session_id=m.session_id,
        user_id=m.user_id,
        role=m.role,
        content=m.content,
        created_at=m.created_at,
        groundedness=m.groundedness,
        relevance=m.relevance,
        confidence=m.confidence,
        flagged=m.flagged,
        eval_reasoning=m.eval_reasoning,
        tools_called=m.tools_called,
    )


def _to_stored_session(s: DBSession) -> StoredSession:
    return StoredSession(
        id=s.id,
        user_id=s.user_id,
        created_at=s.created_at,
        updated_at=s.updated_at,
        messages=[_to_stored_message(m) for m in s.messages],
    )


class SQLiteMemoryBackend(AbstractMemoryBackend):

    def __init__(self, db: AsyncSession):
        self._db = db

    # ── Sessions ──────────────────────────────────────────────────────────────

    async def get_or_create_session(
        self, user_id: str, session_id: Optional[str]
    ) -> str:
        if session_id:
            result = await self._db.execute(
                select(DBSession).where(
                    DBSession.id == session_id,
                    DBSession.user_id == user_id,
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.updated_at = _now()
                return existing.id

        new_session = DBSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
        )
        self._db.add(new_session)
        await self._db.flush()
        return new_session.id

    async def get_sessions_for_user(self, user_id: str) -> list[StoredSession]:
        result = await self._db.execute(
            select(DBSession)
            .where(DBSession.user_id == user_id)
            .options(selectinload(DBSession.messages))
            .order_by(DBSession.created_at.desc())
        )
        sessions = result.scalars().all()
        return [_to_stored_session(s) for s in sessions]

    # ── Messages ──────────────────────────────────────────────────────────────

    async def save_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        eval_data: Optional[dict] = None,
        tools_called: Optional[list[str]] = None,
    ) -> StoredMessage:
        msg = DBMessage(
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            tools_called=tools_called,
        )
        if eval_data:
            msg.groundedness = eval_data.get("groundedness")
            msg.relevance = eval_data.get("relevance")
            msg.confidence = eval_data.get("confidence")
            msg.flagged = eval_data.get("flagged")
            msg.eval_reasoning = eval_data.get("reasoning")

        self._db.add(msg)
        await self._db.flush()
        return _to_stored_message(msg)

    # ── User memory ───────────────────────────────────────────────────────────

    async def get_user_facts(self, user_id: str) -> list[MemoryFact]:
        result = await self._db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == user_id)
            .order_by(UserMemory.created_at)
        )
        rows = result.scalars().all()
        return [
            MemoryFact(
                id=r.id,
                user_id=r.user_id,
                fact_type=r.fact_type,
                content=r.content,
                source_session_id=r.source_session_id,
                created_at=r.created_at,
            )
            for r in rows
        ]

    async def upsert_user_fact(
        self,
        user_id: str,
        content: str,
        fact_type: str = "fact",
        source_session_id: Optional[str] = None,
    ) -> MemoryFact:
        fact = UserMemory(
            user_id=user_id,
            fact_type=fact_type,
            content=content,
            source_session_id=source_session_id,
        )
        self._db.add(fact)
        await self._db.flush()
        return MemoryFact(
            id=fact.id,
            user_id=fact.user_id,
            fact_type=fact.fact_type,
            content=fact.content,
            source_session_id=fact.source_session_id,
            created_at=fact.created_at,
        )

    # ── GDPR delete ───────────────────────────────────────────────────────────

    async def delete_user_data(self, user_id: str) -> dict[str, int]:
        # Count first
        sessions_result = await self._db.execute(
            select(func.count()).where(DBSession.user_id == user_id)
        )
        sessions_count = sessions_result.scalar() or 0

        facts_result = await self._db.execute(
            select(func.count()).where(UserMemory.user_id == user_id)
        )
        facts_count = facts_result.scalar() or 0

        # Delete (messages cascade from sessions)
        await self._db.execute(delete(DBSession).where(DBSession.user_id == user_id))
        await self._db.execute(delete(UserMemory).where(UserMemory.user_id == user_id))

        return {"sessions_deleted": sessions_count, "memory_facts_deleted": facts_count}

    # ── Eval aggregation ──────────────────────────────────────────────────────

    async def get_eval_stats(self, user_id: str) -> dict:
        result = await self._db.execute(
            select(
                func.count(DBMessage.id),
                func.avg(DBMessage.groundedness),
                func.avg(DBMessage.relevance),
                func.avg(DBMessage.confidence),
                func.sum(DBMessage.flagged.cast(type_=None)),
            ).where(
                DBMessage.user_id == user_id,
                DBMessage.role == "assistant",
                DBMessage.confidence.isnot(None),
            )
        )
        row = result.one()
        total = row[0] or 0
        avg_g = round(row[1] or 0.0, 3)
        avg_r = round(row[2] or 0.0, 3)
        avg_c = round(row[3] or 0.0, 3)
        flagged = int(row[4] or 0)

        # High confidence count
        hc_result = await self._db.execute(
            select(func.count()).where(
                DBMessage.user_id == user_id,
                DBMessage.role == "assistant",
                DBMessage.confidence >= 0.80,
            )
        )
        hc_count = hc_result.scalar() or 0
        hc_pct = round((hc_count / total * 100) if total else 0.0, 1)

        return {
            "total_responses": total,
            "avg_groundedness": avg_g,
            "avg_relevance": avg_r,
            "avg_confidence": avg_c,
            "flagged_count": flagged,
            "high_confidence_pct": hc_pct,
        }
