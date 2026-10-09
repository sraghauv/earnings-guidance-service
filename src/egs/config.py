"""Runtime settings, read from the environment or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """EGS_-prefixed environment variables, plus the two vendor API keys.

    The vendor keys keep their conventional names so the Anthropic SDK and any
    other tooling pick up the same variable we do.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="EGS_", extra="ignore"
    )

    transcripts_parquet: Path | None = None
    db_path: Path = Path("data/egs.db")
    cache_dir: Path = Path(".cache")

    # One of: ollama, anthropic, fixture
    llm_provider: str = "ollama"

    anthropic_model: str = "claude-opus-5"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"

    # Up to this many schema-repair round trips before an extraction is failed.
    max_repair_attempts: int = 2

    # A guided figure and the reported actual count as agreeing when they land
    # within this fraction of each other.
    reconcile_tolerance: float = 0.02

    alphavantage_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "ALPHAVANTAGE_API_KEY", "EGS_ALPHAVANTAGE_API_KEY"
        ),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "EGS_ANTHROPIC_API_KEY"),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
