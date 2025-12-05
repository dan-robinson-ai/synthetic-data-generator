"""Base provider interface for AI model providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from agentic_synth.core.config import GeneratorConfig


class BaseProvider(ABC):
    """Abstract base class for AI model providers."""

    def __init__(self, config: GeneratorConfig):
        """
        Initialize the provider.

        Args:
            config: Generator configuration
        """
        self.config = config
        self._client: Any = None

    @property
    def model(self) -> str:
        """Get the model to use."""
        return self.config.get_default_model()

    @property
    @abstractmethod
    def client(self) -> Any:
        """Get or create the API client."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text synchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text asynchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated
        """
        pass

    def generate_with_schema(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate structured output conforming to a schema.

        Default implementation adds schema to prompt. Providers may override
        with native structured output support.

        Args:
            prompt: The user prompt
            schema: JSON schema for the output
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated structured data
        """
        import json

        schema_str = json.dumps(schema, indent=2)
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"Output must be valid JSON conforming to this schema:\n{schema_str}\n\n"
            f"Respond with JSON only, no markdown formatting or explanation."
        )

        response = self.generate(
            prompt=enhanced_prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

        # Clean and parse response
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return json.loads(cleaned.strip())

    def close(self) -> None:
        """Clean up resources."""
        self._client = None

    def __enter__(self) -> "BaseProvider":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
