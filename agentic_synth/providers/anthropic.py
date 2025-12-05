"""Anthropic Claude provider implementation."""

from collections.abc import AsyncGenerator
from typing import Any

import anthropic

from agentic_synth.core.config import GeneratorConfig
from agentic_synth.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""

    def __init__(self, config: GeneratorConfig):
        """
        Initialize the Anthropic provider.

        Args:
            config: Generator configuration
        """
        super().__init__(config)
        self._async_client: anthropic.AsyncAnthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    @property
    def async_client(self) -> anthropic.AsyncAnthropic:
        """Get or create the async Anthropic client."""
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._async_client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using Claude.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.temperature),
        )

        # Extract text from content blocks
        text_parts = []
        for block in message.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return "".join(text_parts)

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text asynchronously using Claude.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        message = await self.async_client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.temperature),
        )

        # Extract text from content blocks
        text_parts = []
        for block in message.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return "".join(text_parts)

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation using Claude.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated
        """
        async with self.async_client.messages.stream(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.temperature),
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def close(self) -> None:
        """Clean up resources."""
        super().close()
        self._async_client = None
