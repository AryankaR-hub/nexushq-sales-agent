"""
flag_for_human tool (bonus)
───────────────────────────
Called by the agent when it cannot answer confidently.
Logs a FlaggedEvent to the DB for human reviewer inspection.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FlaggedEvent


async def flag_for_human(
    user_id: str,
    session_id: str,
    reason: str,
    confidence_score: float,
    db: AsyncSession,
) -> str:
    """
    Escalate a conversation to human review.

    Args:
        user_id:          Current user.
        session_id:       Current session.
        reason:           Why the agent is flagging (low confidence, ambiguity, etc.)
        confidence_score: The agent's self-assessed confidence.
        db:               Database session.

    Returns:
        Confirmation string.
    """
    event = FlaggedEvent(
        user_id=user_id,
        session_id=session_id,
        reason=reason,
        confidence_score=confidence_score,
        resolved=False,
    )
    db.add(event)
    await db.flush()
    return (
        f"Flagged for human review. Reason: {reason}. "
        f"A sales representative will follow up with this user."
    )


# Anthropic tool definition
FLAG_FOR_HUMAN_TOOL = {
    "name": "flag_for_human",
    "description": (
        "Escalate this conversation to a human sales representative. "
        "Call this when: (1) confidence is below 0.60, (2) the user asks something "
        "not covered by the catalog, (3) the user seems ready to buy but has blockers. "
        "After flagging, still provide the best answer you can."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "Current user_id"},
            "reason": {
                "type": "string",
                "description": "Why you're escalating (e.g. 'User asked about HIPAA audit process — beyond catalog scope')",
            },
        },
        "required": ["user_id", "reason"],
    },
}
