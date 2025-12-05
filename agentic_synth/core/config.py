"""Configuration classes for the synthetic data generator."""

from typing import Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import os


class GeneratorConfig(BaseModel):
    """Configuration for the synthetic data generator."""

    provider: Literal["openai", "anthropic", "gemini", "openrouter"] = Field(
        default="openai",
        description="The AI provider to use for generation",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for the provider (falls back to environment variable)",
    )
    model: str | None = Field(
        default=None,
        description="Model to use (uses provider default if not specified)",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (0.0-2.0)",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens for generation",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching for improved performance",
    )
    cache_ttl: int = Field(
        default=3600,
        gt=0,
        description="Cache time-to-live in seconds",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom base URL for the API (useful for OpenRouter)",
    )
    timeout: float = Field(
        default=60.0,
        gt=0,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for failed requests",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v: str | None, info) -> str | None:
        """Resolve API key from environment if not provided."""
        if v is not None:
            return v

        # Get provider from values if available
        provider = info.data.get("provider", "openai") if info.data else "openai"

        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        env_var = env_var_map.get(provider)
        if env_var:
            return os.getenv(env_var)

        return None

    def get_default_model(self) -> str:
        """Get the default model for the configured provider."""
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
            "gemini": "gemini-1.5-flash",
            "openrouter": "anthropic/claude-sonnet-4-20250514",
        }
        return self.model or defaults.get(self.provider, "gpt-4o-mini")


class EnvironmentSettings(BaseSettings):
    """Environment-based settings for the synthetic data generator."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    cache_dir: str = Field(
        default="~/.cache/agentic-synth",
        alias="AGENTIC_SYNTH_CACHE_DIR",
    )
    cache_ttl: int = Field(
        default=3600,
        alias="AGENTIC_SYNTH_CACHE_TTL",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> EnvironmentSettings:
    """Get the environment settings singleton."""
    return EnvironmentSettings()
