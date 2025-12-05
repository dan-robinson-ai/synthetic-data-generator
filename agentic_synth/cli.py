"""Command-line interface for Agentic Synth."""

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agentic_synth import GeneratorConfig, SyntheticGenerator
from agentic_synth.generators import EventConfig, TimeSeriesConfig, EmbeddingsConfig

console = Console()


def output_data(
    data: list[dict[str, Any]],
    output: str | None,
    format: str,
) -> None:
    """Output data to file or stdout."""
    if format == "json":
        content = json.dumps(data, indent=2, default=str)
    elif format == "jsonl":
        content = "\n".join(json.dumps(item, default=str) for item in data)
    elif format == "csv":
        import csv
        import io

        if not data:
            content = ""
        else:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            content = buffer.getvalue()
    else:
        content = json.dumps(data, indent=2, default=str)

    if output:
        Path(output).write_text(content)
        console.print(f"[green]Data written to {output}[/green]")
    else:
        console.print(content)


@click.group()
@click.version_option(version="0.1.0", prog_name="agentic-synth")
def main() -> None:
    """Agentic Synth - AI-powered synthetic data generation."""
    pass


@main.group()
def generate() -> None:
    """Generate synthetic data of various types."""
    pass


@generate.command("structured")
@click.option(
    "--schema",
    "-s",
    required=True,
    help='JSON schema for data (e.g., \'{"name": "string", "email": "email"}\')',
)
@click.option("--count", "-n", default=10, help="Number of records to generate")
@click.option("--context", "-c", default="", help="Context for generation")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "gemini", "openrouter"]),
    default="openai",
    help="AI provider to use",
)
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "jsonl", "csv"]),
    default="json",
    help="Output format",
)
@click.option("--api-key", envvar="API_KEY", default=None, help="API key")
def generate_structured(
    schema: str,
    count: int,
    context: str,
    provider: str,
    model: str | None,
    output: str | None,
    format: str,
    api_key: str | None,
) -> None:
    """Generate structured data based on a schema."""
    try:
        schema_dict = json.loads(schema)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON schema: {e}[/red]")
        sys.exit(1)

    config = GeneratorConfig(
        provider=provider,
        model=model,
        api_key=api_key,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Generating {count} records...", total=None)

        with SyntheticGenerator(config) as generator:
            data = generator.generate_structured(
                schema=schema_dict,
                count=count,
                context=context,
            )

    console.print(f"[green]Generated {len(data)} records[/green]")
    output_data(data, output, format)


@generate.command("time-series")
@click.option("--start", required=True, help="Start time (ISO format)")
@click.option("--end", required=True, help="End time (ISO format)")
@click.option("--frequency", "-f", default="1h", help="Frequency (e.g., 1m, 5m, 1h, 1d)")
@click.option(
    "--columns",
    "-c",
    required=True,
    help="Comma-separated column names",
)
@click.option("--context", default="", help="Context for generation")
@click.option(
    "--trend",
    type=click.Choice(["up", "down", "flat", "cyclical", "random"]),
    default="random",
    help="Overall trend",
)
@click.option(
    "--volatility",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    help="Volatility level",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "gemini", "openrouter"]),
    default="openai",
    help="AI provider to use",
)
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option(
    "--format",
    type=click.Choice(["json", "jsonl", "csv"]),
    default="json",
    help="Output format",
)
@click.option("--api-key", envvar="API_KEY", default=None, help="API key")
def generate_time_series(
    start: str,
    end: str,
    frequency: str,
    columns: str,
    context: str,
    trend: str,
    volatility: str,
    provider: str,
    model: str | None,
    output: str | None,
    format: str,
    api_key: str | None,
) -> None:
    """Generate time series data."""
    config = GeneratorConfig(
        provider=provider,
        model=model,
        api_key=api_key,
    )

    ts_config = TimeSeriesConfig(
        start_time=start,
        end_time=end,
        frequency=frequency,
        columns=columns.split(","),
        context=context,
        trend=trend,
        volatility=volatility,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Generating time series data...", total=None)

        with SyntheticGenerator(config) as generator:
            data = generator.generate_time_series(ts_config)

    console.print(f"[green]Generated {len(data)} data points[/green]")
    output_data(data, output, format)


@generate.command("events")
@click.option(
    "--types",
    "-t",
    required=True,
    help="Comma-separated event types",
)
@click.option("--count", "-n", default=50, help="Number of events to generate")
@click.option("--context", "-c", default="", help="Context for generation")
@click.option("--time-range", default=None, help="Time range (e.g., 'last 24 hours')")
@click.option("--include-metadata/--no-metadata", default=True, help="Include metadata")
@click.option("--include-actor/--no-actor", default=True, help="Include actor info")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "gemini", "openrouter"]),
    default="openai",
    help="AI provider to use",
)
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option(
    "--format",
    type=click.Choice(["json", "jsonl", "csv"]),
    default="json",
    help="Output format",
)
@click.option("--api-key", envvar="API_KEY", default=None, help="API key")
def generate_events(
    types: str,
    count: int,
    context: str,
    time_range: str | None,
    include_metadata: bool,
    include_actor: bool,
    provider: str,
    model: str | None,
    output: str | None,
    format: str,
    api_key: str | None,
) -> None:
    """Generate event data."""
    config = GeneratorConfig(
        provider=provider,
        model=model,
        api_key=api_key,
    )

    event_config = EventConfig(
        event_types=types.split(","),
        count=count,
        context=context,
        time_range=time_range,
        include_metadata=include_metadata,
        include_actor=include_actor,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Generating {count} events...", total=None)

        with SyntheticGenerator(config) as generator:
            data = generator.generate_events(event_config)

    console.print(f"[green]Generated {len(data)} events[/green]")
    output_data(data, output, format)


@generate.command("embeddings")
@click.option(
    "--type",
    "-t",
    "content_type",
    type=click.Choice([
        "qa_pairs", "documents", "paragraphs", "sentences",
        "code_snippets", "product_descriptions", "reviews", "articles",
    ]),
    default="documents",
    help="Type of content to generate",
)
@click.option("--count", "-n", default=50, help="Number of items to generate")
@click.option("--context", "-c", default="", help="Domain context")
@click.option(
    "--embedding-model",
    default="text-embedding-3-small",
    help="Embedding model to use",
)
@click.option("--include-embeddings/--no-embeddings", default=True, help="Generate embeddings")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "gemini", "openrouter"]),
    default="openai",
    help="AI provider to use",
)
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option(
    "--format",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    help="Output format",
)
@click.option("--api-key", envvar="API_KEY", default=None, help="API key")
def generate_embeddings(
    content_type: str,
    count: int,
    context: str,
    embedding_model: str,
    include_embeddings: bool,
    provider: str,
    model: str | None,
    output: str | None,
    format: str,
    api_key: str | None,
) -> None:
    """Generate content with embeddings for RAG systems."""
    config = GeneratorConfig(
        provider=provider,
        model=model,
        api_key=api_key,
    )

    emb_config = EmbeddingsConfig(
        content_type=content_type,
        count=count,
        context=context,
        embedding_model=embedding_model,
        include_embeddings=include_embeddings,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Generating {count} {content_type}...", total=None)

        with SyntheticGenerator(config) as generator:
            data = generator.generate_embeddings(emb_config)

    console.print(f"[green]Generated {len(data)} items[/green]")
    output_data(data, output, format)


@main.command("providers")
def list_providers() -> None:
    """List available AI providers."""
    table = Table(title="Available Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Default Model", style="green")
    table.add_column("Environment Variable", style="yellow")

    providers = [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("anthropic", "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
        ("gemini", "gemini-1.5-flash", "GOOGLE_API_KEY"),
        ("openrouter", "anthropic/claude-sonnet-4-20250514", "OPENROUTER_API_KEY"),
    ]

    for provider, model, env_var in providers:
        table.add_row(provider, model, env_var)

    console.print(table)


@main.command("cache")
@click.option("--clear", is_flag=True, help="Clear the cache")
@click.option("--stats", is_flag=True, help="Show cache statistics")
def cache(clear: bool, stats: bool) -> None:
    """Manage the generation cache."""
    from agentic_synth.core.cache import CacheManager

    cache_manager = CacheManager()

    if clear:
        cache_manager.clear()
        console.print("[green]Cache cleared[/green]")

    if stats:
        cache_stats = cache_manager.stats
        table = Table(title="Cache Statistics")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        for key, value in cache_stats.items():
            table.add_row(str(key), str(value))

        console.print(table)

    cache_manager.close()


if __name__ == "__main__":
    main()
