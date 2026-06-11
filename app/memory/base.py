"""
Abstract memory interface.
Any concrete backend (SQLite, Postgres, Mem0, Redis) must implement this.
Swapping backends = changing ONE import in memory/factory.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MemoryFact:
    id: int
    user_id: str
    fact_type: str       # "fact" | "summary"
    content: str
    source_session_id: Optional[str]
    created_at: datetime


@dataclass
class StoredMessage:
    id: int
    session_id: str
    user_id: str
    role: str            # "user" | "assistant"
    content: str
    created_at: datetime
    groundedness: Optional[float] = None
    relevance: Optional[float] = None
    confidence: Optional[float] = None
    flagged: Optional[bool] = None
    eval_reasoning: Optional[str] = None
    tools_called: Optional[list[str]] = None


@dataclass
class StoredSession:
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[StoredMessage]


class AbstractMemoryBackend(ABC):

    # ── Sessions ──────────────────────────────

    @abstractmethod
    async def get_or_create_session(self, user_id: str, session_id: Optional[str]) -> str:
        """Return session_id (existing or freshly created)."""
        ...

    @abstractmethod
    async def get_sessions_for_user(self, user_id: str) -> list[StoredSession]:
        """All sessions with all messages for a user."""
        ...

    # ── Messages ──────────────────────────────

    @abstractmethod
    async def save_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        eval_data: Optional[dict] = None,
        tools_called: Optional[list[str]] = None,
    ) -> StoredMessage:
        ...

    # ── User memory (cross-session facts) ─────

    @abstractmethod
    async def get_user_facts(self, user_id: str) -> list[MemoryFact]:
        """All persisted facts about a user."""
        ...

    @abstractmethod
    async def upsert_user_fact(
        self,
        user_id: str,
        content: str,
        fact_type: str = "fact",
        source_session_id: Optional[str] = None,
    ) -> MemoryFact:
        ...

    @abstractmethod
    async def delete_user_data(self, user_id: str) -> dict[str, int]:
        """GDPR reset — delete all sessions, messages, and facts for user."""
        ...

    # ── Eval aggregation ──────────────────────

    @abstractmethod
    async def get_eval_stats(self, user_id: str) -> dict:
        ...
