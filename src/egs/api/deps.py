"""FastAPI dependency wiring: DB connection, LLM provider, AlphaVantage client.

The app builds real instances of all three at startup (see api/main.py's
lifespan) and stores them on `app.state`. These functions just read that
state back out, so tests can swap in fakes with `app.dependency_overrides`
without touching the app factory.
"""

import sqlite3
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from egs.config import Settings

if TYPE_CHECKING:
    from egs.crosscheck.alphavantage import AlphaVantageClient
    from egs.llm.base import LLMProvider


def build_provider(settings: Settings) -> "LLMProvider":
    """Pick an LLM backend from `settings.llm_provider`.

    Each provider pulls its own config out of `get_settings()` already, so
    this just has to pick the class. Imports are local to each branch so
    choosing "fixture" never requires the anthropic or ollama SDKs to be
    importable, and vice versa.
    """
    if settings.llm_provider == "anthropic":
        from egs.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if settings.llm_provider == "ollama":
        from egs.llm.ollama_provider import OllamaProvider

        return OllamaProvider()
    if settings.llm_provider == "fixture":
        from egs.llm.fixture_provider import FixtureProvider

        # FixtureProvider falls back to <directory>/<cache_key>.json for
        # anything not in its mapping, so cache_dir doubles as the fixture
        # set -- there's no separate config knob for this.
        return FixtureProvider(settings.cache_dir)
    raise ValueError(f"unknown EGS_LLM_PROVIDER: {settings.llm_provider!r}")


def build_av_client(settings: Settings) -> "AlphaVantageClient":
    """Build the AlphaVantage client, or raise if there's no key configured."""
    if not settings.alphavantage_api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY is not set")
    from egs.crosscheck.alphavantage import AlphaVantageClient

    return AlphaVantageClient(api_key=settings.alphavantage_api_key)


def get_app_settings(request: Request) -> Settings:
    """The Settings create_app() was built with, not the global lru_cache.

    Lets a test pass its own Settings to create_app() and have routes see
    it, instead of every route silently reading the process-wide default.
    """
    return request.app.state.settings


def get_conn(request: Request) -> sqlite3.Connection:
    """Default dependency: the connection opened by the app lifespan."""
    conn = getattr(request.app.state, "conn", None)
    if conn is None:
        raise HTTPException(status_code=503, detail="database connection not ready")
    return conn


def get_provider(request: Request) -> "LLMProvider":
    """Default dependency: the provider built at startup from settings.

    Returns None's absence as a 503 rather than letting callers hit
    AttributeError on a missing provider (e.g. ollama host unreachable).
    """
    provider = getattr(request.app.state, "provider", None)
    if provider is None:
        detail = getattr(request.app.state, "provider_error", "no LLM provider configured")
        raise HTTPException(status_code=503, detail=f"LLM provider unavailable: {detail}")
    return provider


def get_av_client(request: Request) -> "AlphaVantageClient":
    """Default dependency: the AlphaVantage client built at startup.

    Unlike get_provider, missing here is not necessarily fatal to a request —
    routes that can proceed without it should prefer getattr(..., None) over
    depending on this directly.
    """
    client = getattr(request.app.state, "av_client", None)
    if client is None:
        detail = getattr(request.app.state, "av_client_error", "no AlphaVantage key configured")
        raise HTTPException(status_code=503, detail=f"AlphaVantage client unavailable: {detail}")
    return client
