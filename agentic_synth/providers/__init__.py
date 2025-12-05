"""AI provider implementations."""

from agentic_synth.core.config import GeneratorConfig
from agentic_synth.providers.base import BaseProvider
from agentic_synth.providers.openai import OpenAIProvider
from agentic_synth.providers.anthropic import AnthropicProvider
from agentic_synth.providers.google import GeminiProvider
from agentic_synth.providers.openrouter import OpenRouterProvider


def get_provider(config: GeneratorConfig) -> BaseProvider:
    """
    Get the appropriate provider instance based on configuration.

    Args:
        config: Generator configuration

    Returns:
        Provider instance

    Raises:
        ValueError: If the provider is not supported
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "openrouter": OpenRouterProvider,
    }

    provider_class = providers.get(config.provider)
    if provider_class is None:
        raise ValueError(
            f"Unsupported provider: {config.provider}. "
            f"Supported providers: {list(providers.keys())}"
        )

    return provider_class(config)


__all__ = [
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OpenRouterProvider",
    "get_provider",
]
