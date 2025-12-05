"""Basic usage example for Agentic Synth."""

from agentic_synth import GeneratorConfig, SyntheticGenerator


def main() -> None:
    """Demonstrate basic usage of the synthetic data generator."""

    # Initialize with OpenAI (default provider)
    # Make sure to set OPENAI_API_KEY environment variable
    config = GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
    )

    with SyntheticGenerator(config) as generator:
        # Generate user profiles
        print("Generating user profiles...")
        users = generator.generate_structured(
            schema={
                "name": "name",
                "email": "email",
                "age": "integer",
                "city": "city",
                "is_active": "boolean",
            },
            count=5,
            context="Generate realistic user profiles for a SaaS application",
        )

        print("\nGenerated Users:")
        for user in users:
            print(f"  - {user['name']} ({user['email']})")

        # Generate product data
        print("\nGenerating products...")
        products = generator.generate_structured(
            schema={
                "name": "string",
                "description": "paragraph",
                "price": "price",
                "category": "string",
                "in_stock": "boolean",
            },
            count=3,
            context="E-commerce products for a tech store",
        )

        print("\nGenerated Products:")
        for product in products:
            print(f"  - {product['name']}: ${product['price']}")


if __name__ == "__main__":
    main()
