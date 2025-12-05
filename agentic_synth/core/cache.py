"""Caching system for improved performance."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from diskcache import Cache


class CacheManager:
    """Manages caching for synthetic data generation requests."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        ttl: int = 3600,
        enabled: bool = True,
    ):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory for cache storage. Defaults to ~/.cache/agentic-synth
            ttl: Time-to-live for cache entries in seconds
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.ttl = ttl

        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/agentic-synth")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._cache: Cache | None = None
        if enabled:
            self._cache = Cache(str(self.cache_dir))

    def _generate_key(self, **kwargs: Any) -> str:
        """Generate a cache key from the provided arguments."""
        # Sort keys for consistent hashing
        sorted_items = sorted(kwargs.items(), key=lambda x: x[0])
        key_string = json.dumps(sorted_items, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(
        self,
        provider: str,
        model: str,
        prompt: str,
        schema: dict | None = None,
        **kwargs: Any,
    ) -> Any | None:
        """
        Retrieve a cached result.

        Args:
            provider: The AI provider used
            model: The model used
            prompt: The generation prompt
            schema: The data schema (if applicable)
            **kwargs: Additional parameters that affect the result

        Returns:
            The cached result if found, None otherwise
        """
        if not self.enabled or self._cache is None:
            return None

        key = self._generate_key(
            provider=provider,
            model=model,
            prompt=prompt,
            schema=schema,
            **kwargs,
        )

        return self._cache.get(key)

    def set(
        self,
        value: Any,
        provider: str,
        model: str,
        prompt: str,
        schema: dict | None = None,
        ttl: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Store a result in the cache.

        Args:
            value: The value to cache
            provider: The AI provider used
            model: The model used
            prompt: The generation prompt
            schema: The data schema (if applicable)
            ttl: Custom TTL for this entry (uses default if not specified)
            **kwargs: Additional parameters that affect the result
        """
        if not self.enabled or self._cache is None:
            return

        key = self._generate_key(
            provider=provider,
            model=model,
            prompt=prompt,
            schema=schema,
            **kwargs,
        )

        self._cache.set(key, value, expire=ttl or self.ttl)

    def clear(self) -> None:
        """Clear all cached entries."""
        if self._cache is not None:
            self._cache.clear()

    def close(self) -> None:
        """Close the cache connection."""
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __enter__(self) -> "CacheManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._cache is None:
            return {"enabled": False}

        return {
            "enabled": self.enabled,
            "directory": str(self.cache_dir),
            "ttl": self.ttl,
            "size": len(self._cache),
            "volume": self._cache.volume(),
        }
