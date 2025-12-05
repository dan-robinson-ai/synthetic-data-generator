"""Event data generator."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from agentic_synth.generators.base import BaseDataGenerator


class EventConfig(BaseModel):
    """Configuration for event generation."""

    event_types: list[str] = Field(
        description="List of event types to generate",
    )
    count: int = Field(
        default=50,
        gt=0,
        description="Number of events to generate",
    )
    context: str = Field(
        default="",
        description="Additional context for generation",
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in events",
    )
    include_actor: bool = Field(
        default=True,
        description="Include actor/user information",
    )
    time_range: str | None = Field(
        default=None,
        description="Time range for events (e.g., 'last 24 hours', 'last week')",
    )
    severity_distribution: dict[str, float] | None = Field(
        default=None,
        description="Distribution of severity levels (e.g., {'low': 0.6, 'medium': 0.3, 'high': 0.1})",
    )
    correlation_groups: int = Field(
        default=0,
        ge=0,
        description="Number of correlated event groups (events that are related)",
    )


class EventGenerator(BaseDataGenerator):
    """Generator for event data."""

    def generate(
        self,
        config: EventConfig,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate event data.

        Args:
            config: Event configuration
            **kwargs: Additional parameters

        Returns:
            List of generated events
        """
        event_types_str = ", ".join(config.event_types)
        severity_str = ""
        if config.severity_distribution:
            severity_str = f"Severity distribution: {config.severity_distribution}"

        prompt = f"""Generate {config.count} realistic event records with the following specifications:

Event types: {event_types_str}
{f"Context: {config.context}" if config.context else ""}
{f"Time range: {config.time_range}" if config.time_range else ""}
{severity_str}
{f"Create {config.correlation_groups} groups of correlated events" if config.correlation_groups > 0 else ""}

Each event must include:
- event_id: Unique identifier
- event_type: One of the specified types
- timestamp: ISO format timestamp
- description: Brief description of the event
{f"- actor: Object with user_id, username, and role" if config.include_actor else ""}
{f"- metadata: Relevant additional data specific to the event type" if config.include_metadata else ""}
- severity: low, medium, high, or critical
{f"- correlation_id: ID linking related events" if config.correlation_groups > 0 else ""}

Requirements:
- Generate exactly {config.count} unique events
- Distribute event types realistically
- Timestamps should be in chronological order within the time range
- Events should tell a coherent story when applicable

Return a JSON array of event objects. Respond with ONLY the JSON array."""

        # Check cache
        cached = self._check_cache(
            prompt=prompt,
            schema={"event_types": config.event_types},
            config=config.model_dump(),
        )
        if cached is not None:
            return cached

        system_prompt = (
            "You are an event data generator for security, analytics, and system logging. "
            "Generate realistic events that could occur in real-world systems. "
            "Ensure events are coherent, properly timestamped, and include relevant details."
        )

        result = self.provider.generate_with_schema(
            prompt=prompt,
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string"},
                        "event_type": {"type": "string"},
                        "timestamp": {"type": "string"},
                        "description": {"type": "string"},
                        "severity": {"type": "string"},
                    },
                },
            },
            system_prompt=system_prompt,
            **kwargs,
        )

        if isinstance(result, dict):
            result = result.get("data", result.get("events", result.get("items", [result])))

        # Store in cache
        self._store_cache(
            value=result,
            prompt=prompt,
            schema={"event_types": config.event_types},
            config=config.model_dump(),
        )

        return result

    async def stream(
        self,
        config: EventConfig,
        batch_size: int = 10,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream event data generation.

        Args:
            config: Event configuration
            batch_size: Number of events per batch
            **kwargs: Additional parameters

        Yields:
            Events one at a time
        """
        generated = 0
        event_types_str = ", ".join(config.event_types)

        while generated < config.count:
            batch_count = min(batch_size, config.count - generated)

            prompt = f"""Generate {batch_count} realistic event records:

Event types: {event_types_str}
{f"Context: {config.context}" if config.context else ""}

This is batch {generated // batch_size + 1}. Continue the event sequence.

Each event must have: event_id, event_type, timestamp, description, severity
{f"Include actor information" if config.include_actor else ""}
{f"Include relevant metadata" if config.include_metadata else ""}

Return a JSON array of event objects."""

            system_prompt = (
                "You are an event data generator. Generate realistic, coherent events."
            )

            response = await self.provider.agenerate(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            )

            # Parse response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            try:
                batch = json.loads(cleaned.strip())
                if isinstance(batch, dict):
                    batch = batch.get("data", batch.get("events", batch.get("items", [batch])))

                for item in batch:
                    yield item
                    generated += 1
                    if generated >= config.count:
                        break
            except json.JSONDecodeError:
                continue
