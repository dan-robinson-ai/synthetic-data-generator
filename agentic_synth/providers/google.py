"""Google Gemini provider implementation."""

from collections.abc import AsyncGenerator
from typing import Any

import google.generativeai as genai

from agentic_synth.core.config import GeneratorConfig
from agentic_synth.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini models."""

    def __init__(self, config: GeneratorConfig):
        """
        Initialize the Gemini provider.

        Args:
            config: Generator configuration
        """
        super().__init__(config)
        self._configured = False

    def _ensure_configured(self) -> None:
        """Ensure the Gemini API is configured."""
        if not self._configured:
            genai.configure(api_key=self.config.api_key)
            self._configured = True

    @property
    def client(self) -> genai.GenerativeModel:
        """Get or create the Gemini model."""
        if self._client is None:
            self._ensure_configured()
            self._client = genai.GenerativeModel(
                model_name=self.model,
                generation_config=genai.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                ),
            )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using Gemini.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = genai.GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )

        response = self.client.generate_content(
            full_prompt,
            generation_config=generation_config,
        )

        return response.text

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text asynchronously using Gemini.

        Note: Gemini SDK doesn't have native async support,
        so this runs synchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # Gemini doesn't have native async support yet
        # Use sync version
        return self.generate(prompt, system_prompt, **kwargs)

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation using Gemini.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = genai.GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )

        response = self.client.generate_content(
            full_prompt,
            generation_config=generation_config,
            stream=True,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def create_embeddings(
        self,
        texts: list[str],
        model: str = "models/embedding-001",
    ) -> list[list[float]]:
        """
        Create embeddings for the given texts.

        Args:
            texts: List of texts to embed
            model: Embedding model to use

        Returns:
            List of embedding vectors
        """
        self._ensure_configured()

        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])

        return embeddings
