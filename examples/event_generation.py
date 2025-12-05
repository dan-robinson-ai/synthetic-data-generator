"""Event generation example."""

from agentic_synth import GeneratorConfig, SyntheticGenerator
from agentic_synth.generators import EventConfig


def main() -> None:
    """Demonstrate event data generation."""

    config = GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini",
    )

    with SyntheticGenerator(config) as generator:
        # Generate security events
        print("Generating security audit events...")
        security_config = EventConfig(
            event_types=["login", "logout", "password_change", "permission_change", "file_access"],
            count=10,
            context="Enterprise security audit log for a financial institution",
            include_metadata=True,
            include_actor=True,
            time_range="last 24 hours",
            severity_distribution={
                "low": 0.5,
                "medium": 0.3,
                "high": 0.15,
                "critical": 0.05,
            },
        )

        security_events = generator.generate_events(security_config)

        print("\nSecurity Events:")
        for event in security_events[:5]:
            print(
                f"  [{event.get('severity', 'N/A').upper()}] "
                f"{event.get('event_type', 'N/A')}: {event.get('description', 'N/A')[:50]}..."
            )

        # Generate e-commerce events
        print("\nGenerating e-commerce analytics events...")
        ecommerce_config = EventConfig(
            event_types=["page_view", "add_to_cart", "remove_from_cart", "checkout", "purchase"],
            count=15,
            context="E-commerce website user journey events",
            include_metadata=True,
            include_actor=True,
            correlation_groups=3,  # Create 3 user journeys
        )

        ecommerce_events = generator.generate_events(ecommerce_config)

        print("\nE-commerce Events:")
        for event in ecommerce_events[:5]:
            print(
                f"  {event.get('event_type', 'N/A')}: {event.get('description', 'N/A')[:50]}..."
            )


if __name__ == "__main__":
    main()
