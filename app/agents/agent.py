"""
SalesAgent — agentic loop using Gemini with function calling.

Flow per turn:
  1. Load recent session history → build contents list
  2. Send to Gemini with tool declarations
  3. If Gemini returns function_call parts → dispatch to real Python callables
  4. Feed tool results back → Gemini produces final text response
  5. Return (response_text, tools_called, catalog_snippets)
"""

from __future__ import annotations

import logging
from typing import Any

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

from app.agents.prompts import SALES_AGENT_SYSTEM
from app.config import get_settings
from app.memory.base import AbstractMemoryBackend
from app.tools.catalog_tool import search_catalog
from app.tools.flag_tool import flag_for_human
from app.tools.memory_tool import get_user_memory

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Gemini tool declarations ──────────────────────────────────────────────────

_SEARCH_CATALOG_DECL = FunctionDeclaration(
    name="search_catalog",
    description=(
        "Search the NexusHQ product catalog. Use this to answer questions about "
        "plans, pricing, features, limits, add-ons, SSO, compliance, trials, "
        "refunds, or anything product-related. Always call this before answering "
        "pricing or feature questions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query about plans, pricing, or features",
            }
        },
        "required": ["query"],
    },
)

_GET_USER_MEMORY_DECL = FunctionDeclaration(
    name="get_user_memory",
    description=(
        "Retrieve everything known about this user from past conversations. "
        "Call this at the start of every response to load cross-session context — "
        "plan interest, questions already asked, preferences, company size, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "The user_id of the current user",
            }
        },
        "required": ["user_id"],
    },
)

_FLAG_FOR_HUMAN_DECL = FunctionDeclaration(
    name="flag_for_human",
    description=(
        "Escalate this conversation to a human sales representative. "
        "Call this when: (1) confidence is low, (2) user asks something not in catalog, "
        "(3) user seems ready to buy but has blockers. Still answer after flagging."
    ),
    parameters={
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "Current user_id"},
            "reason": {
                "type": "string",
                "description": "Why you are escalating",
            },
        },
        "required": ["user_id", "reason"],
    },
)

GEMINI_TOOLS = [Tool(function_declarations=[
    _GET_USER_MEMORY_DECL,
    _SEARCH_CATALOG_DECL,
    _FLAG_FOR_HUMAN_DECL,
])]


class SalesAgent:
    def __init__(
        self,
        memory: AbstractMemoryBackend,
        db,
        api_key: str,
    ):
        self._memory = memory
        self._db = db
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=settings.agent_model,
            system_instruction=SALES_AGENT_SYSTEM,
            tools=GEMINI_TOOLS,
        )

    async def run(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
    ) -> tuple[str, list[str], str]:
        """
        Execute one agent turn.
        Returns: (response_text, tools_called, catalog_snippets_used)
        """
        import asyncio

        # ── 1. Build history from current session ────────────────────────────
        sessions = await self._memory.get_sessions_for_user(user_id)
        current_session = next((s for s in sessions if s.id == session_id), None)

        history = []
        if current_session:
            for m in current_session.messages:
                role = "user" if m.role == "user" else "model"
                history.append({"role": role, "parts": [m.content]})

        # ── 2. Start chat with history ───────────────────────────────────────
        chat = self._model.start_chat(history=history)

        tools_called: list[str] = []
        catalog_snippets: list[str] = []

        # ── 3. Agentic tool-use loop ─────────────────────────────────────────
        current_message = user_message

        while True:
            # Run sync Gemini call in thread pool to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda msg=current_message: chat.send_message(msg)
            )

            # Check if any function calls in this response
            function_calls = []
            text_parts = []

            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call.name:
                    function_calls.append(part.function_call)
                elif hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)

            # If no function calls → final text response
            if not function_calls:
                final_text = " ".join(text_parts).strip()
                return final_text or "I'm sorry, I couldn't process that request.", tools_called, "\n\n".join(catalog_snippets)

            # ── 4. Dispatch all function calls ───────────────────────────────
            import google.generativeai.protos as protos
            tool_response_parts = []

            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args)
                tools_called.append(tool_name)
                logger.info("Tool called: %s | args: %s", tool_name, tool_args)

                if tool_name == "search_catalog":
                    result = search_catalog(tool_args.get("query", ""))
                    catalog_snippets.append(result)

                elif tool_name == "get_user_memory":
                    result = await get_user_memory(
                        tool_args.get("user_id", user_id), self._memory
                    )

                elif tool_name == "flag_for_human":
                    result = await flag_for_human(
                        user_id=user_id,
                        session_id=session_id,
                        reason=tool_args.get("reason", ""),
                        confidence_score=0.0,
                        db=self._db,
                    )
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_response_parts.append(
                    protos.Part(
                        function_response=protos.FunctionResponse(
                            name=tool_name,
                            response={"result": result},
                        )
                    )
                )

            # Feed results back — use a content object
            current_message = tool_response_parts