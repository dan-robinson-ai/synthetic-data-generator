"""Tests for main SyntheticGenerator class."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from agentic_synth import GeneratorConfig, SyntheticGenerator
from agentic_synth.generators import TimeSeriesConfig, EventConfig, EmbeddingsConfig


class TestSyntheticGenerator:
    """Tests for SyntheticGenerator."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        generator = SyntheticGenerator()

        assert generator.config.provider == "openai"
        assert generator.config.cache_enabled is True
        generator.close()

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = GeneratorConfig(
            provider="anthropic",
            api_key="test-key",
            cache_enabled=False,
        )
        generator = SyntheticGenerator(config)

        assert generator.config.provider == "anthropic"
        assert generator.config.cache_enabled is False
        generator.close()

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        config = GeneratorConfig(api_key="test-key")

        with SyntheticGenerator(config) as generator:
            assert generator is not None
            assert isinstance(generator, SyntheticGenerator)

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GeneratorConfig(api_key="test-key")
            generator = SyntheticGenerator(config)

            stats = generator.cache_stats
            assert "enabled" in stats
            generator.close()

    def test_clear_cache(self) -> None:
        """Test clearing cache."""
        config = GeneratorConfig(api_key="test-key")
        generator = SyntheticGenerator(config)

        # Should not raise
        generator.clear_cache()
        generator.close()

    @patch("agentic_synth.providers.openai.OpenAI")
    def test_provider_lazy_loading(self, mock_openai: MagicMock) -> None:
        """Test that provider is lazily loaded."""
        config = GeneratorConfig(api_key="test-key")
        generator = SyntheticGenerator(config)

        # Provider should not be created yet
        assert generator._provider is None

        # Accessing provider creates it
        _ = generator.provider
        assert generator._provider is not None

        generator.close()


class TestStructuredDataGeneration:
    """Tests for structured data generation."""

    def test_schema_types(self) -> None:
        """Test that various schema types are valid."""
        from agentic_synth.generators.structured import StructuredDataGenerator

        generator = StructuredDataGenerator.__new__(StructuredDataGenerator)

        schema = {
            "name": "string",
            "age": "integer",
            "email": "email",
            "is_active": "boolean",
        }

        prompt = generator._build_schema_prompt(schema)

        assert "name" in prompt
        assert "text string" in prompt
        assert "integer number" in prompt
        assert "email address" in prompt
        assert "true or false" in prompt


class TestTimeSeriesGeneration:
    """Tests for time series generation."""

    def test_frequency_parsing(self) -> None:
        """Test frequency parsing."""
        from agentic_synth.generators.time_series import TimeSeriesGenerator
        from datetime import timedelta

        generator = TimeSeriesGenerator.__new__(TimeSeriesGenerator)

        assert generator._get_frequency_delta("1h") == timedelta(hours=1)
        assert generator._get_frequency_delta("1d") == timedelta(days=1)
        assert generator._get_frequency_delta("5m") == timedelta(minutes=5)
        assert generator._get_frequency_delta("30s") == timedelta(seconds=30)

    def test_point_calculation(self) -> None:
        """Test data point calculation."""
        from agentic_synth.generators.time_series import TimeSeriesGenerator

        generator = TimeSeriesGenerator.__new__(TimeSeriesGenerator)

        config = TimeSeriesConfig(
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T12:00:00",
            frequency="1h",
        )

        points = generator._calculate_points(config)
        assert points == 13  # 12 hours + 1 for start


class TestEventGeneration:
    """Tests for event generation."""

    def test_event_config_validation(self) -> None:
        """Test event configuration validation."""
        config = EventConfig(
            event_types=["login", "logout"],
            count=10,
        )

        assert len(config.event_types) == 2
        assert config.count == 10


class TestEmbeddingsGeneration:
    """Tests for embeddings generation."""

    def test_content_templates(self) -> None:
        """Test that all content types have templates."""
        from agentic_synth.generators.embeddings import EmbeddingsGenerator

        generator = EmbeddingsGenerator.__new__(EmbeddingsGenerator)

        content_types = [
            "qa_pairs",
            "documents",
            "paragraphs",
            "sentences",
            "code_snippets",
            "product_descriptions",
            "reviews",
            "articles",
        ]

        for content_type in content_types:
            assert content_type in generator.CONTENT_TEMPLATES
