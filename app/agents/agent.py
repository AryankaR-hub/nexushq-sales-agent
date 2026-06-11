"""
SalesAgent — agentic loop with tool use.

Flow per turn:
  1. Load recent session history → build messages list
  2. Send to Claude with tool definitions (get_user_memory, search_catalog, flag_for_human)
  3. If Claude returns tool_use blocks → dispatch to real Python callables
  4. Feed tool results back → Claude produces final text response
  5. Return (response_text, tools_called, catalog_snippets)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from app.agents.prompts import SALES_AGENT_SYSTEM
from app.config import get_settings
from app.memory.base import AbstractMemoryBackend
from app.tools.catalog_tool import SEARCH_CATALOG_TOOL, search_catalog
from app.tools.flag_tool import FLAG_FOR_HUMAN_TOOL, flag_for_human
from app.tools.memory_tool import GET_USER_MEMORY_TOOL, get_user_memory

logger = logging.getLogger(__name__)
settings = get_settings()

ALL_TOOLS = [GET_USER_MEMORY_TOOL, SEARCH_CATALOG_TOOL, FLAG_FOR_HUMAN_TOOL]


class SalesAgent:
    def __init__(
        self,
        memory: AbstractMemoryBackend,
        db,                    # AsyncSession — needed for flag_for_human
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        self._memory = memory
        self._db = db
        self._client = anthropic_client

    async def run(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
    ) -> tuple[str, list[str], str]:
        """
        Execute one agent turn.

        Returns:
            (response_text, tools_called, catalog_snippets_used)
        """
        # ── 1. Build message history from current session ────────────────────
        sessions = await self._memory.get_sessions_for_user(user_id)
        current_session = next(
            (s for s in sessions if s.id == session_id), None
        )

        messages: list[dict[str, Any]] = []
        if current_session:
            for m in current_session.messages:
                messages.append({"role": m.role, "content": m.content})

        # Append current user message
        messages.append({"role": "user", "content": user_message})

        # ── 2. Agentic tool-use loop ─────────────────────────────────────────
        tools_called: list[str] = []
        catalog_snippets: list[str] = []
        flag_invoked = False

        while True:
            response = await self._client.messages.create(
                model=settings.agent_model,
                max_tokens=1024,
                system=SALES_AGENT_SYSTEM,
                tools=ALL_TOOLS,
                messages=messages,
            )

            # Append assistant response to messages for multi-turn tool loop
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Final text response
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text = block.text
                        break
                return text, tools_called, "\n\n".join(catalog_snippets)

            if response.stop_reason == "tool_use":
                # Dispatch every tool call in this response
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_name = block.name
                    tool_input = block.input
                    tools_called.append(tool_name)
                    logger.info("Tool called: %s | input: %s", tool_name, tool_input)

                    # ── Dispatch ──────────────────────────────────────────────
                    if tool_name == "search_catalog":
                        result = search_catalog(tool_input.get("query", ""))
                        catalog_snippets.append(result)

                    elif tool_name == "get_user_memory":
                        result = await get_user_memory(
                            tool_input.get("user_id", user_id), self._memory
                        )

                    elif tool_name == "flag_for_human":
                        result = await flag_for_human(
                            user_id=user_id,
                            session_id=session_id,
                            reason=tool_input.get("reason", ""),
                            confidence_score=0.0,  # pre-eval; updated after eval
                            db=self._db,
                        )
                        flag_invoked = True

                    else:
                        result = f"Unknown tool: {tool_name}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason
                logger.warning("Unexpected stop_reason: %s", response.stop_reason)
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text = block.text
                        break
                return text or "I'm sorry, I couldn't process that request.", tools_called, ""
