"""Example using multiple AI providers."""

from agentic_synth import GeneratorConfig, SyntheticGenerator


def generate_with_provider(provider: str, model: str | None = None) -> None:
    """Generate data using a specific provider."""
    print(f"\n{'=' * 50}")
    print(f"Using provider: {provider}")
    print(f"Model: {model or 'default'}")
    print("=" * 50)

    config = GeneratorConfig(
        provider=provider,
        model=model,
        temperature=0.7,
    )

    try:
        with SyntheticGenerator(config) as generator:
            data = generator.generate_structured(
                schema={
                    "product_name": "string",
                    "category": "string",
                    "price": "price",
                    "rating": "float",
                },
                count=3,
                context="E-commerce product listings",
            )

            print("\nGenerated Products:")
            for item in data:
                print(f"  - {item.get('product_name', 'N/A')}: ${item.get('price', 'N/A')}")

    except Exception as e:
        print(f"  Error: {e}")


def main() -> None:
    """Demonstrate using different AI providers."""

    # List of providers to try
    providers = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("gemini", "gemini-1.5-flash"),
        ("openrouter", "anthropic/claude-sonnet-4-20250514"),
    ]

    print("Multi-Provider Synthetic Data Generation Example")
    print("Make sure to set the appropriate API keys in environment variables:")
    print("  - OPENAI_API_KEY")
    print("  - ANTHROPIC_API_KEY")
    print("  - GOOGLE_API_KEY")
    print("  - OPENROUTER_API_KEY")

    for provider, model in providers:
        generate_with_provider(provider, model)


if __name__ == "__main__":
    main()
