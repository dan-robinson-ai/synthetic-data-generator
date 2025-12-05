"""Base generator interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from agentic_synth.core.cache import CacheManager
from agentic_synth.providers.base import BaseProvider


class BaseDataGenerator(ABC):
    """Abstract base class for data generators."""

    def __init__(
        self,
        provider: BaseProvider,
        cache: CacheManager | None = None,
    ):
        """
        Initialize the generator.

        Args:
            provider: AI provider for generation
            cache: Optional cache manager
        """
        self.provider = provider
        self.cache = cache

    @abstractmethod
    def generate(self, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Generate data synchronously.

        Returns:
            List of generated data items
        """
        pass

    @abstractmethod
    async def stream(self, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream data generation.

        Yields:
            Generated data items one at a time
        """
        pass

    def _check_cache(
        self,
        prompt: str,
        schema: dict | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]] | None:
        """Check for cached results."""
        if self.cache is None:
            return None

        return self.cache.get(
            provider=self.provider.config.provider,
            model=self.provider.model,
            prompt=prompt,
            schema=schema,
            **kwargs,
        )

    def _store_cache(
        self,
        value: list[dict[str, Any]],
        prompt: str,
        schema: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Store results in cache."""
        if self.cache is None:
            return

        self.cache.set(
            value=value,
            provider=self.provider.config.provider,
            model=self.provider.model,
            prompt=prompt,
            schema=schema,
            **kwargs,
        )
