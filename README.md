# Agentic Synth (Python)

AI-powered synthetic data generation for ML training, RAG systems, and testing workflows.

A Python port of [@ruvector/agentic-synth](https://www.npmjs.com/package/@ruvector/agentic-synth).

## Features

- **Multi-model support**: Integrates with OpenAI GPT, Claude, Gemini, and 50+ models via OpenRouter
- **Data generation types**: Time-series, events, structured data, and embeddings
- **Performance optimization**: Context caching for improved performance
- **Streaming capability**: AsyncGenerator for real-time data flow without memory constraints
- **Self-learning**: Quality optimization through feedback loops
- **Type-safe**: Full Pydantic validation and type hints

## Installation

```bash
pip install agentic-synth
```

Or install from source:

```bash
git clone https://github.com/synthetic-data-generator/agentic-synth.git
cd agentic-synth
pip install -e .
```

## Quick Start

### Basic Usage

```python
from agentic_synth import SyntheticGenerator, GeneratorConfig

# Initialize with your preferred provider
config = GeneratorConfig(
    provider="openai",
    api_key="your-api-key",
    model="gpt-4o-mini"
)

generator = SyntheticGenerator(config)

# Generate structured data
data = generator.generate_structured(
    schema={
        "name": "string",
        "age": "integer",
        "email": "email",
        "is_active": "boolean"
    },
    count=10,
    context="Generate realistic user profiles for a SaaS application"
)

for item in data:
    print(item)
```

### Streaming Generation

```python
import asyncio
from agentic_synth import SyntheticGenerator, GeneratorConfig

async def stream_data():
    config = GeneratorConfig(
        provider="anthropic",
        api_key="your-api-key",
        model="claude-sonnet-4-20250514"
    )

    generator = SyntheticGenerator(config)

    async for item in generator.stream_structured(
        schema={"event": "string", "timestamp": "datetime", "value": "float"},
        count=100,
        context="IoT sensor data stream"
    ):
        print(item)

asyncio.run(stream_data())
```

### Time Series Generation

```python
from agentic_synth import SyntheticGenerator, GeneratorConfig
from agentic_synth.generators import TimeSeriesConfig

config = GeneratorConfig(provider="openai", api_key="your-key")
generator = SyntheticGenerator(config)

# Generate stock price time series
time_series = generator.generate_time_series(
    TimeSeriesConfig(
        start_time="2024-01-01",
        end_time="2024-12-31",
        frequency="1h",
        columns=["open", "high", "low", "close", "volume"],
        context="Stock market trading data for a tech company"
    )
)
```

### Event Generation

```python
from agentic_synth import SyntheticGenerator, GeneratorConfig
from agentic_synth.generators import EventConfig

config = GeneratorConfig(provider="gemini", api_key="your-key")
generator = SyntheticGenerator(config)

# Generate security events
events = generator.generate_events(
    EventConfig(
        event_types=["login", "logout", "file_access", "permission_change"],
        count=50,
        context="Security audit log for enterprise application",
        include_metadata=True
    )
)
```

### Embeddings Generation

```python
from agentic_synth import SyntheticGenerator, GeneratorConfig
from agentic_synth.generators import EmbeddingsConfig

config = GeneratorConfig(provider="openai", api_key="your-key")
generator = SyntheticGenerator(config)

# Generate text with embeddings for RAG
embeddings = generator.generate_embeddings(
    EmbeddingsConfig(
        content_type="qa_pairs",
        count=100,
        context="Customer support FAQ for e-commerce platform",
        embedding_model="text-embedding-3-small"
    )
)
```

## CLI Usage

```bash
# Generate structured data
agentic-synth generate structured \
    --schema '{"name": "string", "email": "email"}' \
    --count 10 \
    --provider openai \
    --output users.json

# Generate time series
agentic-synth generate time-series \
    --start "2024-01-01" \
    --end "2024-12-31" \
    --frequency "1d" \
    --columns "price,volume" \
    --output stock_data.csv

# Generate events
agentic-synth generate events \
    --types "click,purchase,view" \
    --count 100 \
    --output events.json
```

## Configuration

### Environment Variables

```bash
# Provider API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...

# Optional settings
AGENTIC_SYNTH_CACHE_DIR=~/.cache/agentic-synth
AGENTIC_SYNTH_CACHE_TTL=3600
```

### Provider Options

| Provider | Models | Features |
|----------|--------|----------|
| `openai` | gpt-4o, gpt-4o-mini, gpt-4-turbo | Function calling, JSON mode |
| `anthropic` | claude-sonnet-4-20250514, claude-3-5-haiku | Long context, structured output |
| `gemini` | gemini-1.5-pro, gemini-1.5-flash | Multimodal, caching |
| `openrouter` | 50+ models | Model routing, fallbacks |

## API Reference

### SyntheticGenerator

The main class for generating synthetic data.

```python
class SyntheticGenerator:
    def __init__(self, config: GeneratorConfig): ...

    # Synchronous methods
    def generate_structured(self, schema: dict, count: int, context: str) -> list[dict]: ...
    def generate_time_series(self, config: TimeSeriesConfig) -> list[dict]: ...
    def generate_events(self, config: EventConfig) -> list[dict]: ...
    def generate_embeddings(self, config: EmbeddingsConfig) -> list[dict]: ...

    # Async streaming methods
    async def stream_structured(self, schema: dict, count: int, context: str) -> AsyncGenerator: ...
    async def stream_time_series(self, config: TimeSeriesConfig) -> AsyncGenerator: ...
    async def stream_events(self, config: EventConfig) -> AsyncGenerator: ...
```

### GeneratorConfig

```python
class GeneratorConfig:
    provider: Literal["openai", "anthropic", "gemini", "openrouter"]
    api_key: str | None = None  # Falls back to environment variable
    model: str | None = None  # Uses provider default if not specified
    temperature: float = 0.7
    max_tokens: int = 4096
    cache_enabled: bool = True
    cache_ttl: int = 3600
```

## Examples

The `examples/` directory contains comprehensive examples:

- `basic_usage.py` - Getting started
- `time_series_example.py` - Financial and IoT time series
- `event_generation.py` - Security and analytics events
- `structured_data.py` - User profiles, products, transactions
- `streaming_example.py` - Real-time data generation
- `rag_embeddings.py` - RAG system data preparation

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see LICENSE file for details.

## Credits

Python port inspired by [@ruvector/agentic-synth](https://www.npmjs.com/package/@ruvector/agentic-synth).
