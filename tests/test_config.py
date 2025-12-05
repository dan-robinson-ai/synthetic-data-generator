"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from agentic_synth.core.config import GeneratorConfig, get_settings


class TestGeneratorConfig:
    """Tests for GeneratorConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = GeneratorConfig()

        assert config.provider == "openai"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.cache_enabled is True
        assert config.cache_ttl == 3600
        assert config.timeout == 60.0
        assert config.max_retries == 3

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = GeneratorConfig(
            provider="anthropic",
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            temperature=0.5,
            max_tokens=2048,
        )

        assert config.provider == "anthropic"
        assert config.api_key == "test-key"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_get_default_model_openai(self) -> None:
        """Test default model for OpenAI."""
        config = GeneratorConfig(provider="openai")
        assert config.get_default_model() == "gpt-4o-mini"

    def test_get_default_model_anthropic(self) -> None:
        """Test default model for Anthropic."""
        config = GeneratorConfig(provider="anthropic")
        assert config.get_default_model() == "claude-sonnet-4-20250514"

    def test_get_default_model_gemini(self) -> None:
        """Test default model for Gemini."""
        config = GeneratorConfig(provider="gemini")
        assert config.get_default_model() == "gemini-1.5-flash"

    def test_get_default_model_openrouter(self) -> None:
        """Test default model for OpenRouter."""
        config = GeneratorConfig(provider="openrouter")
        assert config.get_default_model() == "anthropic/claude-sonnet-4-20250514"

    def test_custom_model_overrides_default(self) -> None:
        """Test that custom model overrides default."""
        config = GeneratorConfig(provider="openai", model="gpt-4-turbo")
        assert config.get_default_model() == "gpt-4-turbo"

    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        with pytest.raises(ValueError):
            GeneratorConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            GeneratorConfig(temperature=2.1)

    def test_max_tokens_validation(self) -> None:
        """Test max_tokens validation."""
        with pytest.raises(ValueError):
            GeneratorConfig(max_tokens=0)

        with pytest.raises(ValueError):
            GeneratorConfig(max_tokens=-1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    def test_api_key_from_environment(self) -> None:
        """Test API key resolution from environment."""
        config = GeneratorConfig(provider="openai")
        # Note: The validator runs during initialization
        # API key should be resolved from environment

    def test_invalid_provider(self) -> None:
        """Test invalid provider raises error."""
        with pytest.raises(ValueError):
            GeneratorConfig(provider="invalid_provider")


class TestEnvironmentSettings:
    """Tests for environment settings."""

    def test_get_settings(self) -> None:
        """Test getting environment settings."""
        settings = get_settings()
        assert settings is not None
        assert settings.cache_ttl == 3600
