"""Time series data generator."""

import json
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_synth.generators.base import BaseDataGenerator


class TimeSeriesConfig(BaseModel):
    """Configuration for time series generation."""

    start_time: str | datetime = Field(
        description="Start time for the series (ISO format string or datetime)"
    )
    end_time: str | datetime = Field(
        description="End time for the series (ISO format string or datetime)"
    )
    frequency: str = Field(
        default="1h",
        description="Frequency of data points (e.g., '1m', '5m', '1h', '1d')",
    )
    columns: list[str] = Field(
        default_factory=lambda: ["value"],
        description="Column names to generate",
    )
    context: str = Field(
        default="",
        description="Additional context for generation",
    )
    trend: Literal["up", "down", "flat", "cyclical", "random"] = Field(
        default="random",
        description="Overall trend of the time series",
    )
    volatility: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Volatility of the values",
    )
    include_anomalies: bool = Field(
        default=False,
        description="Whether to include anomalous data points",
    )
    anomaly_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Rate of anomalies (0.0-1.0)",
    )


class TimeSeriesGenerator(BaseDataGenerator):
    """Generator for time series data."""

    FREQUENCY_MAP = {
        "1s": timedelta(seconds=1),
        "5s": timedelta(seconds=5),
        "10s": timedelta(seconds=10),
        "30s": timedelta(seconds=30),
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "10m": timedelta(minutes=10),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "2h": timedelta(hours=2),
        "4h": timedelta(hours=4),
        "6h": timedelta(hours=6),
        "12h": timedelta(hours=12),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
    }

    def _parse_time(self, time_val: str | datetime) -> datetime:
        """Parse a time value to datetime."""
        if isinstance(time_val, datetime):
            return time_val
        return datetime.fromisoformat(time_val.replace("Z", "+00:00"))

    def _get_frequency_delta(self, frequency: str) -> timedelta:
        """Get timedelta from frequency string."""
        if frequency in self.FREQUENCY_MAP:
            return self.FREQUENCY_MAP[frequency]

        # Try to parse custom frequency
        import re

        match = re.match(r"(\d+)([smhdw])", frequency)
        if match:
            value, unit = match.groups()
            value = int(value)
            units = {
                "s": timedelta(seconds=value),
                "m": timedelta(minutes=value),
                "h": timedelta(hours=value),
                "d": timedelta(days=value),
                "w": timedelta(weeks=value),
            }
            return units.get(unit, timedelta(hours=1))

        return timedelta(hours=1)

    def _calculate_points(self, config: TimeSeriesConfig) -> int:
        """Calculate the number of data points."""
        start = self._parse_time(config.start_time)
        end = self._parse_time(config.end_time)
        delta = self._get_frequency_delta(config.frequency)
        return int((end - start) / delta) + 1

    def generate(
        self,
        config: TimeSeriesConfig,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate time series data.

        Args:
            config: Time series configuration
            **kwargs: Additional parameters

        Returns:
            List of time series data points
        """
        num_points = self._calculate_points(config)
        start = self._parse_time(config.start_time)
        delta = self._get_frequency_delta(config.frequency)

        prompt = f"""Generate a time series dataset with the following specifications:

- Number of data points: {num_points}
- Start time: {start.isoformat()}
- Frequency: {config.frequency}
- Columns: {", ".join(config.columns)}
- Trend: {config.trend}
- Volatility: {config.volatility}
{f"- Include anomalies: Yes (rate: {config.anomaly_rate * 100:.1f}%)" if config.include_anomalies else ""}
{f"Context: {config.context}" if config.context else ""}

Requirements:
- Generate exactly {num_points} data points
- Each point must have a 'timestamp' field (ISO format) and all specified columns
- Values should follow the specified trend and volatility
- Ensure realistic patterns and correlations between columns
{"- Include some anomalous values marked with an 'is_anomaly' boolean field" if config.include_anomalies else ""}

Return a JSON array of objects. Respond with ONLY the JSON array."""

        # Check cache
        cached = self._check_cache(
            prompt=prompt,
            schema={"columns": config.columns},
            config=config.model_dump(),
        )
        if cached is not None:
            return cached

        system_prompt = (
            "You are a time series data generator. Generate realistic time series data "
            "with appropriate patterns, trends, and correlations. Ensure the data follows "
            "the specified characteristics while maintaining realism."
        )

        result = self.provider.generate_with_schema(
            prompt=prompt,
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string"},
                        **{col: {"type": "number"} for col in config.columns},
                    },
                },
            },
            system_prompt=system_prompt,
            **kwargs,
        )

        if isinstance(result, dict):
            result = result.get("data", result.get("items", [result]))

        # Store in cache
        self._store_cache(
            value=result,
            prompt=prompt,
            schema={"columns": config.columns},
            config=config.model_dump(),
        )

        return result

    async def stream(
        self,
        config: TimeSeriesConfig,
        batch_size: int = 50,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream time series data generation.

        Args:
            config: Time series configuration
            batch_size: Number of points per batch
            **kwargs: Additional parameters

        Yields:
            Time series data points one at a time
        """
        num_points = self._calculate_points(config)
        start = self._parse_time(config.start_time)
        delta = self._get_frequency_delta(config.frequency)

        generated = 0
        current_time = start

        while generated < num_points:
            batch_count = min(batch_size, num_points - generated)
            batch_start = current_time

            prompt = f"""Generate {batch_count} consecutive time series data points:

- Start time: {batch_start.isoformat()}
- Frequency: {config.frequency}
- Columns: {", ".join(config.columns)}
- Trend: {config.trend}
- Volatility: {config.volatility}
{f"Context: {config.context}" if config.context else ""}

This is batch {generated // batch_size + 1}. Continue the pattern from previous batches.

Return a JSON array of objects with 'timestamp' and all column fields."""

            system_prompt = (
                "You are a time series data generator. Generate realistic, "
                "continuous time series data."
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
                    batch = batch.get("data", batch.get("items", [batch]))

                for item in batch:
                    yield item
                    generated += 1
                    current_time += delta
                    if generated >= num_points:
                        break
            except json.JSONDecodeError:
                continue
