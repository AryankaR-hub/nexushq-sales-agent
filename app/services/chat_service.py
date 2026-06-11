"""
ChatService
───────────
Orchestrates: memory → agent → eval → memory write → response

This is the single public interface the API routes call.
"""

from __future__ import annotations

import json
import logging
import re

import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.agent import SalesAgent
from app.agents.prompts import MEMORY_EXTRACTION_PROMPT, MEMORY_SUMMARIZATION_PROMPT
from app.config import get_settings
from app.memory.base import AbstractMemoryBackend
from app.memory.factory import get_memory_backend
from app.models.schemas import ChatResponse, EvalBlock
from app.services.eval_service import EvalService

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatService:
    def __init__(self, db: AsyncSession):
        self._db = db
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._memory: AbstractMemoryBackend = get_memory_backend(db)
        self._agent = SalesAgent(self._memory, db, self._client)
        self._eval = EvalService(self._client)

    async def chat(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None,
    ) -> ChatResponse:
        # ── 1. Get/create session ────────────────────────────────────────────
        active_session_id = await self._memory.get_or_create_session(
            user_id, session_id
        )

        # ── 2. Persist user message ──────────────────────────────────────────
        await self._memory.save_message(
            session_id=active_session_id,
            user_id=user_id,
            role="user",
            content=user_message,
        )

        # ── 3. Run agent ─────────────────────────────────────────────────────
        response_text, tools_called, catalog_snippets = await self._agent.run(
            user_id=user_id,
            session_id=active_session_id,
            user_message=user_message,
        )

        # ── 4. Self-evaluate ─────────────────────────────────────────────────
        eval_data = await self._eval.score(
            user_message=user_message,
            assistant_response=response_text,
            tools_called=tools_called,
            catalog_used=catalog_snippets,
        )

        # ── 5. Persist assistant message + eval ──────────────────────────────
        await self._memory.save_message(
            session_id=active_session_id,
            user_id=user_id,
            role="assistant",
            content=response_text,
            eval_data=eval_data,
            tools_called=tools_called,
        )

        # ── 6. Extract and store new user facts (async background-style) ──────
        await self._extract_and_store_facts(
            user_id=user_id,
            session_id=active_session_id,
            user_message=user_message,
            assistant_message=response_text,
        )

        # ── 7. Compress memory if threshold exceeded ──────────────────────────
        await self._maybe_compress_memory(user_id)

        # ── 8. Build response ─────────────────────────────────────────────────
        return ChatResponse(
            response=response_text,
            eval=EvalBlock(**eval_data),
            tools_called=tools_called,
            session_id=active_session_id,
            user_id=user_id,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _extract_and_store_facts(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Extract atomic facts from this turn and persist them."""
        try:
            prompt = MEMORY_EXTRACTION_PROMPT.format(
                user_message=user_message,
                assistant_message=assistant_message,
            )
            resp = await self._client.messages.create(
                model=settings.agent_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            facts: list[str] = json.loads(raw)
            for fact in facts:
                if isinstance(fact, str) and fact.strip():
                    await self._memory.upsert_user_fact(
                        user_id=user_id,
                        content=fact.strip(),
                        fact_type="fact",
                        source_session_id=session_id,
                    )
        except Exception as e:
            logger.warning("Fact extraction failed (non-fatal): %s", e)

    async def _maybe_compress_memory(self, user_id: str) -> None:
        """
        Bonus: memory summarization.
        When atomic fact count exceeds the threshold, compress into one summary row.
        """
        facts = await self._memory.get_user_facts(user_id)
        atomic_facts = [f for f in facts if f.fact_type == "fact"]

        if len(atomic_facts) < settings.memory_summary_threshold:
            return

        logger.info(
            "Compressing %d facts for user %s", len(atomic_facts), user_id
        )
        try:
            facts_text = "\n".join(f"- {f.content}" for f in atomic_facts)
            prompt = MEMORY_SUMMARIZATION_PROMPT.format(facts=facts_text)
            resp = await self._client.messages.create(
                model=settings.agent_model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = resp.content[0].text.strip()

            # Delete old atomic facts and replace with summary
            from sqlalchemy import delete as sqla_delete
            from app.db.models import UserMemory
            ids_to_delete = [f.id for f in atomic_facts]
            await self._db.execute(
                sqla_delete(UserMemory).where(UserMemory.id.in_(ids_to_delete))
            )
            await self._memory.upsert_user_fact(
                user_id=user_id,
                content=summary,
                fact_type="summary",
            )
            logger.info("Memory compressed for user %s", user_id)
        except Exception as e:
            logger.warning("Memory compression failed (non-fatal): %s", e)
