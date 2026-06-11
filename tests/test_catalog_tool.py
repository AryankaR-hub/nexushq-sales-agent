"""Unit tests for search_catalog tool — no LLM calls needed."""

import pytest
from app.tools.catalog_tool import search_catalog


def test_search_enterprise_pricing():
    result = search_catalog("enterprise pricing SSO")
    assert "Enterprise" in result
    assert "SSO" in result or "499" in result


def test_search_starter_plan():
    result = search_catalog("starter plan features")
    assert "Starter" in result


def test_search_trial():
    result = search_catalog("free trial")
    assert "14-day" in result or "trial" in result.lower()


def test_search_no_match_returns_fallback():
    result = search_catalog("xyzzy123nonexistent")
    assert "Available plans" in result


def test_search_refund():
    result = search_catalog("refund policy money back")
    assert "30-day" in result or "refund" in result.lower()
