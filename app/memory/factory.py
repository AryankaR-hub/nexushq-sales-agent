"""
Memory backend factory.
To swap SQLite → Postgres: change DATABASE_URL in .env (same SQLAlchemy driver).
To swap to Mem0/Redis: implement AbstractMemoryBackend and change the import below.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from app.memory.base import AbstractMemoryBackend
from app.memory.sqlite_backend import SQLiteMemoryBackend   # ← only line to change


def get_memory_backend(db: AsyncSession) -> AbstractMemoryBackend:
    return SQLiteMemoryBackend(db)
