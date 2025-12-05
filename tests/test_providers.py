"""Tests for AI providers."""

import pytest

from agentic_synth.core.config import GeneratorConfig
from agentic_synth.providers import get_provider
from agentic_synth.providers.openai import OpenAIProvider
from agentic_synth.providers.anthropic import AnthropicProvider
from agentic_synth.providers.google import GeminiProvider
from agentic_synth.providers.openrouter import OpenRouterProvider


class TestGetProvider:
    """Tests for get_provider function."""

    def test_get_openai_provider(self) -> None:
        """Test getting OpenAI provider."""
        config = GeneratorConfig(provider="openai", api_key="test-key")
        provider = get_provider(config)

        assert isinstance(provider, OpenAIProvider)
        assert provider.config.provider == "openai"

    def test_get_anthropic_provider(self) -> None:
        """Test getting Anthropic provider."""
        config = GeneratorConfig(provider="anthropic", api_key="test-key")
        provider = get_provider(config)

        assert isinstance(provider, AnthropicProvider)
        assert provider.config.provider == "anthropic"

    def test_get_gemini_provider(self) -> None:
        """Test getting Gemini provider."""
        config = GeneratorConfig(provider="gemini", api_key="test-key")
        provider = get_provider(config)

        assert isinstance(provider, GeminiProvider)
        assert provider.config.provider == "gemini"

    def test_get_openrouter_provider(self) -> None:
        """Test getting OpenRouter provider."""
        config = GeneratorConfig(provider="openrouter", api_key="test-key")
        provider = get_provider(config)

        assert isinstance(provider, OpenRouterProvider)
        assert provider.config.provider == "openrouter"


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_model_property(self) -> None:
        """Test model property."""
        config = GeneratorConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-4-turbo",
        )
        provider = OpenAIProvider(config)

        assert provider.model == "gpt-4-turbo"

    def test_default_model(self) -> None:
        """Test default model."""
        config = GeneratorConfig(provider="openai", api_key="test-key")
        provider = OpenAIProvider(config)

        assert provider.model == "gpt-4o-mini"


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_model_property(self) -> None:
        """Test model property."""
        config = GeneratorConfig(
            provider="anthropic",
            api_key="test-key",
            model="claude-3-opus-20240229",
        )
        provider = AnthropicProvider(config)

        assert provider.model == "claude-3-opus-20240229"

    def test_default_model(self) -> None:
        """Test default model."""
        config = GeneratorConfig(provider="anthropic", api_key="test-key")
        provider = AnthropicProvider(config)

        assert provider.model == "claude-sonnet-4-20250514"


class TestGeminiProvider:
    """Tests for Gemini provider."""

    def test_model_property(self) -> None:
        """Test model property."""
        config = GeneratorConfig(
            provider="gemini",
            api_key="test-key",
            model="gemini-1.5-pro",
        )
        provider = GeminiProvider(config)

        assert provider.model == "gemini-1.5-pro"

    def test_default_model(self) -> None:
        """Test default model."""
        config = GeneratorConfig(provider="gemini", api_key="test-key")
        provider = GeminiProvider(config)

        assert provider.model == "gemini-1.5-flash"


class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""

    def test_model_property(self) -> None:
        """Test model property."""
        config = GeneratorConfig(
            provider="openrouter",
            api_key="test-key",
            model="openai/gpt-4-turbo",
        )
        provider = OpenRouterProvider(config)

        assert provider.model == "openai/gpt-4-turbo"

    def test_default_model(self) -> None:
        """Test default model."""
        config = GeneratorConfig(provider="openrouter", api_key="test-key")
        provider = OpenRouterProvider(config)

        assert provider.model == "anthropic/claude-sonnet-4-20250514"

    def test_base_url(self) -> None:
        """Test base URL."""
        config = GeneratorConfig(provider="openrouter", api_key="test-key")
        provider = OpenRouterProvider(config)

        assert provider.BASE_URL == "https://openrouter.ai/api/v1"
