"""Streaming generation example."""

import asyncio

from agentic_synth import GeneratorConfig, SyntheticGenerator


async def stream_structured_data() -> None:
    """Stream structured data generation."""
    config = GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini",
    )

    generator = SyntheticGenerator(config)

    print("Streaming user data generation...")
    print("-" * 50)

    count = 0
    async for user in generator.stream_structured(
        schema={
            "id": "uuid",
            "name": "name",
            "email": "email",
            "department": "string",
            "hire_date": "date",
        },
        count=10,
        context="Company employee records",
    ):
        count += 1
        print(f"[{count}] {user.get('name', 'N/A')} - {user.get('department', 'N/A')}")

    print("-" * 50)
    print(f"Total generated: {count} records")

    generator.close()


async def stream_events() -> None:
    """Stream event generation."""
    from agentic_synth.generators import EventConfig

    config = GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini",
    )

    generator = SyntheticGenerator(config)

    event_config = EventConfig(
        event_types=["click", "scroll", "form_submit", "page_leave"],
        count=15,
        context="Web analytics events",
    )

    print("\nStreaming web analytics events...")
    print("-" * 50)

    count = 0
    async for event in generator.stream_events(event_config):
        count += 1
        print(f"[{count}] {event.get('event_type', 'N/A')}: {event.get('timestamp', 'N/A')}")

    print("-" * 50)
    print(f"Total generated: {count} events")

    generator.close()


async def main() -> None:
    """Run all streaming examples."""
    await stream_structured_data()
    await stream_events()


if __name__ == "__main__":
    asyncio.run(main())
