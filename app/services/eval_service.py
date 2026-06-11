"""
EvalService
───────────
Prompts an LLM to self-score every assistant response.
Scores are structured, always present, and persisted in the DB.

Limitations acknowledged in README:
- Self-scoring by the same model that generated the answer has a positivity bias.
- At scale, replace with a separate judge model (e.g. GPT-4 or Prometheus).
"""

from __future__ import annotations

import json
import logging
import re

import anthropic

from app.agents.prompts import EVAL_PROMPT
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_DEFAULT_EVAL = {
    "groundedness": 0.5,
    "relevance": 0.5,
    "confidence": 0.5,
    "flagged": True,
    "reasoning": "Eval could not be computed; defaulting to cautious scores.",
}


class EvalService:
    def __init__(self, client: anthropic.AsyncAnthropic):
        self._client = client

    async def score(
        self,
        user_message: str,
        assistant_response: str,
        tools_called: list[str],
        catalog_used: str,
    ) -> dict:
        """
        Returns eval dict with keys: groundedness, relevance, confidence,
        flagged, reasoning.
        """
        prompt = EVAL_PROMPT.format(
            user_message=user_message,
            assistant_response=assistant_response,
            tools_called=", ".join(tools_called) if tools_called else "none",
            catalog_used=catalog_used[:800] if catalog_used else "none",
        )

        try:
            response = await self._client.messages.create(
                model=settings.eval_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip ```json fences if present
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)

            # Clamp floats to [0, 1]
            for key in ("groundedness", "relevance", "confidence"):
                parsed[key] = max(0.0, min(1.0, float(parsed.get(key, 0.5))))

            # Enforce flagged logic: auto-flag if any score < threshold
            threshold = settings.confidence_flag_threshold
            auto_flag = any(
                parsed[k] < threshold for k in ("groundedness", "relevance", "confidence")
            )
            parsed["flagged"] = bool(parsed.get("flagged", False)) or auto_flag

            return parsed

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Eval parsing failed: %s", e)
            return _DEFAULT_EVAL
        except anthropic.APIError as e:
            logger.error("Eval API error: %s", e)
            return _DEFAULT_EVAL
