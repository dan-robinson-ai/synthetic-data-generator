"""Structured data generator."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from agentic_synth.generators.base import BaseDataGenerator


class StructuredDataGenerator(BaseDataGenerator):
    """Generator for structured data based on schemas."""

    # Type mapping for schema types
    TYPE_DESCRIPTIONS = {
        "string": "a text string",
        "integer": "an integer number",
        "int": "an integer number",
        "float": "a floating-point number",
        "number": "a number (integer or float)",
        "boolean": "true or false",
        "bool": "true or false",
        "email": "a valid email address",
        "url": "a valid URL",
        "date": "a date in YYYY-MM-DD format",
        "datetime": "a datetime in ISO 8601 format",
        "time": "a time in HH:MM:SS format",
        "uuid": "a UUID string",
        "phone": "a phone number",
        "address": "a street address",
        "name": "a person's full name",
        "first_name": "a person's first name",
        "last_name": "a person's last name",
        "company": "a company name",
        "job_title": "a job title",
        "paragraph": "a paragraph of text",
        "sentence": "a single sentence",
        "word": "a single word",
        "country": "a country name",
        "city": "a city name",
        "currency": "a currency code (e.g., USD)",
        "price": "a price value with 2 decimal places",
        "percentage": "a percentage value (0-100)",
        "latitude": "a latitude coordinate",
        "longitude": "a longitude coordinate",
        "ip_address": "an IP address",
        "mac_address": "a MAC address",
        "color": "a color name or hex code",
        "array": "an array of items",
        "object": "a nested object",
    }

    def _build_schema_prompt(self, schema: dict[str, str]) -> str:
        """Build a description of the schema for the prompt."""
        fields = []
        for field_name, field_type in schema.items():
            description = self.TYPE_DESCRIPTIONS.get(
                field_type.lower(), f"a value of type {field_type}"
            )
            fields.append(f'  - "{field_name}": {description}')

        return "\n".join(fields)

    def generate(
        self,
        schema: dict[str, str],
        count: int = 10,
        context: str = "",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate structured data based on a schema.

        Args:
            schema: Dictionary mapping field names to types
            count: Number of records to generate
            context: Additional context for generation
            **kwargs: Additional parameters

        Returns:
            List of generated records
        """
        schema_description = self._build_schema_prompt(schema)

        prompt = f"""Generate {count} realistic synthetic data records with the following fields:
{schema_description}

{f"Context: {context}" if context else ""}

Requirements:
- Generate exactly {count} unique records
- Each record must have all specified fields
- Values must be realistic and consistent
- Ensure variety in the generated data

Return a JSON array of objects. Respond with ONLY the JSON array, no explanation."""

        # Check cache
        cached = self._check_cache(
            prompt=prompt,
            schema=schema,
            count=count,
            context=context,
        )
        if cached is not None:
            return cached

        # Generate
        system_prompt = (
            "You are a synthetic data generator. Generate realistic, diverse data "
            "that could plausibly exist in real-world applications. Ensure consistency "
            "and variety in the generated data."
        )

        result = self.provider.generate_with_schema(
            prompt=prompt,
            schema={"type": "array", "items": {"type": "object", "properties": schema}},
            system_prompt=system_prompt,
            **kwargs,
        )

        # Ensure result is a list
        if isinstance(result, dict):
            result = result.get("data", result.get("items", [result]))

        # Store in cache
        self._store_cache(
            value=result,
            prompt=prompt,
            schema=schema,
            count=count,
            context=context,
        )

        return result

    async def stream(
        self,
        schema: dict[str, str],
        count: int = 10,
        context: str = "",
        batch_size: int = 5,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream structured data generation.

        Args:
            schema: Dictionary mapping field names to types
            count: Total number of records to generate
            context: Additional context for generation
            batch_size: Number of records to generate per batch
            **kwargs: Additional parameters

        Yields:
            Generated records one at a time
        """
        generated = 0
        schema_description = self._build_schema_prompt(schema)

        while generated < count:
            batch_count = min(batch_size, count - generated)

            prompt = f"""Generate {batch_count} realistic synthetic data records with the following fields:
{schema_description}

{f"Context: {context}" if context else ""}

This is batch {generated // batch_size + 1}. Generate unique records different from previous batches.

Return a JSON array of objects. Respond with ONLY the JSON array, no explanation."""

            system_prompt = (
                "You are a synthetic data generator. Generate realistic, diverse data "
                "that could plausibly exist in real-world applications."
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
                    if generated >= count:
                        break
            except json.JSONDecodeError:
                # If parsing fails, try to generate one at a time
                continue
