"""OpenRouter provider implementation for accessing 50+ models."""

from collections.abc import AsyncGenerator
from typing import Any

import httpx

from agentic_synth.core.config import GeneratorConfig
from agentic_synth.providers.base import BaseProvider


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter - access to 50+ AI models."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: GeneratorConfig):
        """
        Initialize the OpenRouter provider.

        Args:
            config: Generator configuration
        """
        super().__init__(config)
        self._async_client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url or self.BASE_URL,
                headers=self._get_headers(),
                timeout=self.config.timeout,
            )
        return self._client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.config.base_url or self.BASE_URL,
                headers=self._get_headers(),
                timeout=self.config.timeout,
            )
        return self._async_client

    def _get_headers(self) -> dict[str, str]:
        """Get the request headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/synthetic-data-generator",
            "X-Title": "Agentic Synth",
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using OpenRouter.

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

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text asynchronously using OpenRouter.

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

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        response = await self.async_client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation using OpenRouter.

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

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        async with self.async_client.stream(
            "POST",
            "/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    import json

                    try:
                        chunk = json.loads(data)
                        if chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
                    except json.JSONDecodeError:
                        continue

    def list_models(self) -> list[dict[str, Any]]:
        """
        List available models on OpenRouter.

        Returns:
            List of model information dictionaries
        """
        response = self.client.get("/models")
        response.raise_for_status()
        return response.json()["data"]

    async def alist_models(self) -> list[dict[str, Any]]:
        """
        List available models asynchronously.

        Returns:
            List of model information dictionaries
        """
        response = await self.async_client.get("/models")
        response.raise_for_status()
        return response.json()["data"]

    def close(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.close()
        if self._async_client:
            # Note: async client should be closed in async context
            pass
        super().close()
        self._async_client = None
