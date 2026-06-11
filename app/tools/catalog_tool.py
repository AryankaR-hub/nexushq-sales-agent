"""
search_catalog tool
───────────────────
Performs keyword + fuzzy search over catalog.json.
Returns matching plans, add-ons, and FAQ entries as structured text.
The agent calls this instead of answering from LLM knowledge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_CATALOG_PATH = Path(__file__).parents[2] / "catalog.json"
_catalog: dict[str, Any] | None = None


def _load_catalog() -> dict[str, Any]:
    global _catalog
    if _catalog is None:
        with open(_CATALOG_PATH) as f:
            _catalog = json.load(f)
    return _catalog


def _score_text(text: str, tokens: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for t in tokens if t in text_lower)


def search_catalog(query: str) -> str:
    """
    Search the product catalog for plans, features, pricing, and FAQ.

    Args:
        query: Natural-language query about pricing, features, plans, etc.

    Returns:
        Formatted string with relevant catalog sections.
    """
    catalog = _load_catalog()
    tokens = re.findall(r"\w+", query.lower())
    results: list[tuple[int, str]] = []

    # Score each plan
    for plan in catalog.get("plans", []):
        plan_text = f"{plan['name']} {plan['price']} {' '.join(plan['features'])}"
        plan_text += f" {plan.get('ideal_for', '')} {json.dumps(plan.get('limits', {}))}"
        score = _score_text(plan_text, tokens)
        if score > 0:
            features_str = "\n    - ".join(plan["features"])
            limits = plan.get("limits", {})
            block = (
                f"PLAN: {plan['name']}\n"
                f"  Price: {plan['price']} | Annual: {plan.get('annual_price', 'N/A')}\n"
                f"  Features:\n    - {features_str}\n"
                f"  API calls/mo: {limits.get('api_calls_per_month', 'N/A')}\n"
                f"  Data retention: {limits.get('data_retention_days', 'N/A')} days\n"
                f"  Ideal for: {plan.get('ideal_for', 'N/A')}"
            )
            results.append((score, block))

    # Score each FAQ
    for faq in catalog.get("faq", []):
        faq_text = f"{faq['question']} {faq['answer']}"
        score = _score_text(faq_text, tokens)
        if score > 0:
            block = f"FAQ: {faq['question']}\n  Answer: {faq['answer']}"
            results.append((score, block))

    # Score add-ons
    for addon in catalog.get("add_ons", []):
        addon_text = f"{addon['name']} {addon['price']}"
        score = _score_text(addon_text, tokens)
        if score > 0:
            block = f"ADD-ON: {addon['name']} — {addon['price']}"
            results.append((score, block))

    if not results:
        # Fallback: return all plans in brief
        brief = []
        for plan in catalog.get("plans", []):
            brief.append(f"- {plan['name']}: {plan['price']}")
        return "No specific match found. Available plans:\n" + "\n".join(brief)

    # Sort by relevance score descending, return top 5
    results.sort(key=lambda x: x[0], reverse=True)
    top = [r[1] for r in results[:5]]
    return "\n\n".join(top)


# Anthropic tool definition
SEARCH_CATALOG_TOOL = {
    "name": "search_catalog",
    "description": (
        "Search the NexusHQ product catalog. Use this to answer questions about "
        "plans, pricing, features, limits, add-ons, SSO, compliance, trials, "
        "refunds, or anything product-related. Always call this before answering "
        "pricing or feature questions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query about plans, pricing, or features",
            }
        },
        "required": ["query"],
    },
}
