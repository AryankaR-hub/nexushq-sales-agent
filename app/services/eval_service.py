"""
EvalService — self-scores every assistant response using Gemini.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

import google.generativeai as genai

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
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=settings.eval_model)

    async def score(
        self,
        user_message: str,
        assistant_response: str,
        tools_called: list[str],
        catalog_used: str,
    ) -> dict:
        prompt = EVAL_PROMPT.format(
            user_message=user_message,
            assistant_response=assistant_response,
            tools_called=", ".join(tools_called) if tools_called else "none",
            catalog_used=catalog_used[:800] if catalog_used else "none",
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._model.generate_content(prompt)
            )
            raw = response.text.strip()

            # Strip ```json fences
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)

            for key in ("groundedness", "relevance", "confidence"):
                parsed[key] = max(0.0, min(1.0, float(parsed.get(key, 0.5))))

            threshold = settings.confidence_flag_threshold
            auto_flag = any(
                parsed[k] < threshold for k in ("groundedness", "relevance", "confidence")
            )
            parsed["flagged"] = bool(parsed.get("flagged", False)) or auto_flag
            return parsed

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Eval parsing failed: %s", e)
            return _DEFAULT_EVAL
        except Exception as e:
            logger.error("Eval error: %s", e)
            return _DEFAULT_EVAL