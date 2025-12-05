"""Main synthetic data generator class."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from agentic_synth.core.cache import CacheManager
from agentic_synth.core.config import GeneratorConfig
from agentic_synth.generators.embeddings import EmbeddingsConfig, EmbeddingsGenerator
from agentic_synth.generators.events import EventConfig, EventGenerator
from agentic_synth.generators.structured import StructuredDataGenerator
from agentic_synth.generators.time_series import TimeSeriesConfig, TimeSeriesGenerator
from agentic_synth.providers import get_provider
from agentic_synth.providers.base import BaseProvider


class SyntheticGenerator:
    """
    Main class for generating synthetic data using AI models.

    Supports multiple AI providers (OpenAI, Anthropic, Gemini, OpenRouter)
    and various data types (structured, time-series, events, embeddings).
    """

    def __init__(self, config: GeneratorConfig | None = None):
        """
        Initialize the synthetic data generator.

        Args:
            config: Generator configuration. Uses defaults if not provided.
        """
        self.config = config or GeneratorConfig()
        self._provider: BaseProvider | None = None
        self._cache = CacheManager(
            enabled=self.config.cache_enabled,
            ttl=self.config.cache_ttl,
        )

        # Initialize specialized generators
        self._structured_generator: StructuredDataGenerator | None = None
        self._time_series_generator: TimeSeriesGenerator | None = None
        self._event_generator: EventGenerator | None = None
        self._embeddings_generator: EmbeddingsGenerator | None = None

    @property
    def provider(self) -> BaseProvider:
        """Get or create the AI provider instance."""
        if self._provider is None:
            self._provider = get_provider(self.config)
        return self._provider

    @property
    def structured_generator(self) -> StructuredDataGenerator:
        """Get the structured data generator."""
        if self._structured_generator is None:
            self._structured_generator = StructuredDataGenerator(
                provider=self.provider,
                cache=self._cache,
            )
        return self._structured_generator

    @property
    def time_series_generator(self) -> TimeSeriesGenerator:
        """Get the time series generator."""
        if self._time_series_generator is None:
            self._time_series_generator = TimeSeriesGenerator(
                provider=self.provider,
                cache=self._cache,
            )
        return self._time_series_generator

    @property
    def event_generator(self) -> EventGenerator:
        """Get the event generator."""
        if self._event_generator is None:
            self._event_generator = EventGenerator(
                provider=self.provider,
                cache=self._cache,
            )
        return self._event_generator

    @property
    def embeddings_generator(self) -> EmbeddingsGenerator:
        """Get the embeddings generator."""
        if self._embeddings_generator is None:
            self._embeddings_generator = EmbeddingsGenerator(
                provider=self.provider,
                cache=self._cache,
            )
        return self._embeddings_generator

    # ==================== Structured Data Generation ====================

    def generate_structured(
        self,
        schema: dict[str, str],
        count: int = 10,
        context: str = "",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate structured data based on a schema.

        Args:
            schema: Dictionary mapping field names to types
                   (e.g., {"name": "string", "age": "integer", "email": "email"})
            count: Number of records to generate
            context: Additional context for generation
            **kwargs: Additional generation parameters

        Returns:
            List of generated records matching the schema
        """
        return self.structured_generator.generate(
            schema=schema,
            count=count,
            context=context,
            **kwargs,
        )

    async def stream_structured(
        self,
        schema: dict[str, str],
        count: int = 10,
        context: str = "",
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream structured data generation.

        Args:
            schema: Dictionary mapping field names to types
            count: Number of records to generate
            context: Additional context for generation
            **kwargs: Additional generation parameters

        Yields:
            Generated records one at a time
        """
        async for item in self.structured_generator.stream(
            schema=schema,
            count=count,
            context=context,
            **kwargs,
        ):
            yield item

    # ==================== Time Series Generation ====================

    def generate_time_series(
        self,
        config: TimeSeriesConfig,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate time series data.

        Args:
            config: Time series configuration
            **kwargs: Additional generation parameters

        Returns:
            List of time series data points
        """
        return self.time_series_generator.generate(config=config, **kwargs)

    async def stream_time_series(
        self,
        config: TimeSeriesConfig,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream time series data generation.

        Args:
            config: Time series configuration
            **kwargs: Additional generation parameters

        Yields:
            Time series data points one at a time
        """
        async for item in self.time_series_generator.stream(config=config, **kwargs):
            yield item

    # ==================== Event Generation ====================

    def generate_events(
        self,
        config: EventConfig,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate event data.

        Args:
            config: Event configuration
            **kwargs: Additional generation parameters

        Returns:
            List of generated events
        """
        return self.event_generator.generate(config=config, **kwargs)

    async def stream_events(
        self,
        config: EventConfig,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream event data generation.

        Args:
            config: Event configuration
            **kwargs: Additional generation parameters

        Yields:
            Events one at a time
        """
        async for item in self.event_generator.stream(config=config, **kwargs):
            yield item

    # ==================== Embeddings Generation ====================

    def generate_embeddings(
        self,
        config: EmbeddingsConfig,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate text content with embeddings.

        Args:
            config: Embeddings configuration
            **kwargs: Additional generation parameters

        Returns:
            List of content with embeddings
        """
        return self.embeddings_generator.generate(config=config, **kwargs)

    async def stream_embeddings(
        self,
        config: EmbeddingsConfig,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream embeddings generation.

        Args:
            config: Embeddings configuration
            **kwargs: Additional generation parameters

        Yields:
            Content with embeddings one at a time
        """
        async for item in self.embeddings_generator.stream(config=config, **kwargs):
            yield item

    # ==================== Raw Generation ====================

    def generate_raw(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate raw text output from the AI model.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        return self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def stream_raw(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream raw text generation.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated
        """
        async for chunk in self.provider.stream(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        ):
            yield chunk

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | list[Any]:
        """
        Generate JSON output from the AI model.

        Args:
            prompt: The user prompt
            schema: Optional JSON schema for validation
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Parsed JSON output
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that generates valid JSON output."

        if schema:
            system_prompt += f"\n\nOutput must conform to this schema:\n{json.dumps(schema, indent=2)}"

        response = self.provider.generate(
            prompt=prompt + "\n\nRespond with valid JSON only, no markdown or explanation.",
            system_prompt=system_prompt,
            **kwargs,
        )

        # Clean up response and parse JSON
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return json.loads(cleaned.strip())

    # ==================== Utility Methods ====================

    def clear_cache(self) -> None:
        """Clear the generation cache."""
        self._cache.clear()

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats

    def close(self) -> None:
        """Clean up resources."""
        self._cache.close()
        if self._provider:
            self._provider.close()

    def __enter__(self) -> "SyntheticGenerator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
