"""OpenAI provider implementation."""

from collections.abc import AsyncGenerator
from typing import Any

from openai import OpenAI, AsyncOpenAI

from agentic_synth.core.config import GeneratorConfig
from agentic_synth.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI GPT models."""

    def __init__(self, config: GeneratorConfig):
        """
        Initialize the OpenAI provider.

        Args:
            config: Generator configuration
        """
        super().__init__(config)
        self._async_client: AsyncOpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Get or create the async OpenAI client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
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
        Generate text using OpenAI.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **{k: v for k, v in kwargs.items() if k not in ("temperature", "max_tokens")},
        )

        return response.choices[0].message.content or ""

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text asynchronously using OpenAI.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **{k: v for k, v in kwargs.items() if k not in ("temperature", "max_tokens")},
        )

        return response.choices[0].message.content or ""

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation using OpenAI.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
            **{k: v for k, v in kwargs.items() if k not in ("temperature", "max_tokens")},
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_with_schema(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate structured output using OpenAI's JSON mode.

        Args:
            prompt: The user prompt
            schema: JSON schema for the output
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated structured data
        """
        import json

        messages = []

        schema_str = json.dumps(schema, indent=2)
        enhanced_system = (
            (system_prompt or "You are a helpful assistant.")
            + f"\n\nOutput must be valid JSON conforming to this schema:\n{schema_str}"
        )

        messages.append({"role": "system", "content": enhanced_system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            response_format={"type": "json_object"},
            **{k: v for k, v in kwargs.items() if k not in ("temperature", "max_tokens")},
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def create_embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """
        Create embeddings for the given texts.

        Args:
            texts: List of texts to embed
            model: Embedding model to use

        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    async def acreate_embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """
        Create embeddings asynchronously.

        Args:
            texts: List of texts to embed
            model: Embedding model to use

        Returns:
            List of embedding vectors
        """
        response = await self.async_client.embeddings.create(
            model=model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def close(self) -> None:
        """Clean up resources."""
        super().close()
        self._async_client = None
