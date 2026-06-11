"""
get_user_memory tool
─────────────────────
Retrieves persisted facts about the user from the DB.
This is injected as a real callable tool — not string-stuffed into the system prompt.
The agent calls this on every turn to load cross-session context.
"""

from __future__ import annotations

from app.memory.base import AbstractMemoryBackend


async def get_user_memory(user_id: str, memory: AbstractMemoryBackend) -> str:
    """
    Retrieve all persisted facts and conversation summaries for a user.

    Args:
        user_id: The user whose memory to retrieve.
        memory:  Injected memory backend (not exposed to the LLM tool schema).

    Returns:
        Formatted string of known facts, or a 'no memory' message.
    """
    facts = await memory.get_user_facts(user_id)
    if not facts:
        return "No prior memory found for this user. This appears to be their first interaction."

    lines = [f"Known facts about user '{user_id}':"]
    summaries = [f for f in facts if f.fact_type == "summary"]
    atomic = [f for f in facts if f.fact_type == "fact"]

    if summaries:
        lines.append("\n[Conversation Summary]")
        for s in summaries:
            lines.append(f"  {s.content}")

    if atomic:
        lines.append("\n[Individual Facts]")
        for f in atomic:
            lines.append(f"  - {f.content}")

    return "\n".join(lines)


# Anthropic tool definition
GET_USER_MEMORY_TOOL = {
    "name": "get_user_memory",
    "description": (
        "Retrieve everything known about this user from past conversations. "
        "Call this at the start of every response to load cross-session context — "
        "plan interest, questions already asked, preferences, company size, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "The user_id of the current user",
            }
        },
        "required": ["user_id"],
    },
}
