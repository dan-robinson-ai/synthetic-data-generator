"""Tests for caching module."""

import tempfile
from pathlib import Path

import pytest

from agentic_synth.core.cache import CacheManager


class TestCacheManager:
    """Tests for CacheManager."""

    def test_init_creates_directory(self) -> None:
        """Test that initialization creates cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            cache = CacheManager(cache_dir=cache_dir)

            assert cache_dir.exists()
            cache.close()

    def test_set_and_get(self) -> None:
        """Test setting and getting cached values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            test_data = [{"name": "test", "value": 123}]
            cache.set(
                value=test_data,
                provider="openai",
                model="gpt-4",
                prompt="test prompt",
            )

            result = cache.get(
                provider="openai",
                model="gpt-4",
                prompt="test prompt",
            )

            assert result == test_data
            cache.close()

    def test_cache_miss(self) -> None:
        """Test cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            result = cache.get(
                provider="openai",
                model="gpt-4",
                prompt="nonexistent prompt",
            )

            assert result is None
            cache.close()

    def test_disabled_cache(self) -> None:
        """Test disabled cache doesn't store or retrieve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir, enabled=False)

            test_data = [{"name": "test"}]
            cache.set(
                value=test_data,
                provider="openai",
                model="gpt-4",
                prompt="test prompt",
            )

            result = cache.get(
                provider="openai",
                model="gpt-4",
                prompt="test prompt",
            )

            assert result is None
            cache.close()

    def test_clear_cache(self) -> None:
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            cache.set(
                value=[{"test": 1}],
                provider="openai",
                model="gpt-4",
                prompt="test",
            )

            cache.clear()

            result = cache.get(
                provider="openai",
                model="gpt-4",
                prompt="test",
            )

            assert result is None
            cache.close()

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with CacheManager(cache_dir=tmpdir) as cache:
                cache.set(
                    value=[{"test": 1}],
                    provider="openai",
                    model="gpt-4",
                    prompt="test",
                )
                result = cache.get(
                    provider="openai",
                    model="gpt-4",
                    prompt="test",
                )
                assert result is not None

    def test_stats(self) -> None:
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            stats = cache.stats
            assert stats["enabled"] is True
            assert "directory" in stats
            assert "ttl" in stats
            assert "size" in stats

            cache.close()

    def test_different_params_different_keys(self) -> None:
        """Test that different parameters produce different cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            cache.set(
                value=[{"data": "first"}],
                provider="openai",
                model="gpt-4",
                prompt="test prompt",
            )

            cache.set(
                value=[{"data": "second"}],
                provider="openai",
                model="gpt-4",
                prompt="different prompt",
            )

            result1 = cache.get(
                provider="openai",
                model="gpt-4",
                prompt="test prompt",
            )

            result2 = cache.get(
                provider="openai",
                model="gpt-4",
                prompt="different prompt",
            )

            assert result1 == [{"data": "first"}]
            assert result2 == [{"data": "second"}]
            cache.close()
