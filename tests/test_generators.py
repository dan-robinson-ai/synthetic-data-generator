"""Tests for data generators."""

from datetime import datetime

import pytest

from agentic_synth.generators.time_series import TimeSeriesConfig, TimeSeriesGenerator
from agentic_synth.generators.events import EventConfig
from agentic_synth.generators.embeddings import EmbeddingsConfig


class TestTimeSeriesConfig:
    """Tests for TimeSeriesConfig."""

    def test_basic_config(self) -> None:
        """Test basic time series configuration."""
        config = TimeSeriesConfig(
            start_time="2024-01-01",
            end_time="2024-01-31",
            frequency="1d",
            columns=["value"],
        )

        assert config.frequency == "1d"
        assert config.columns == ["value"]
        assert config.trend == "random"
        assert config.volatility == "medium"

    def test_full_config(self) -> None:
        """Test full time series configuration."""
        config = TimeSeriesConfig(
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T12:00:00",
            frequency="1h",
            columns=["open", "high", "low", "close"],
            context="Stock data",
            trend="up",
            volatility="high",
            include_anomalies=True,
            anomaly_rate=0.1,
        )

        assert config.trend == "up"
        assert config.volatility == "high"
        assert config.include_anomalies is True
        assert config.anomaly_rate == 0.1

    def test_datetime_input(self) -> None:
        """Test datetime objects as input."""
        config = TimeSeriesConfig(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
        )

        assert isinstance(config.start_time, datetime)

    def test_anomaly_rate_validation(self) -> None:
        """Test anomaly rate validation."""
        with pytest.raises(ValueError):
            TimeSeriesConfig(
                start_time="2024-01-01",
                end_time="2024-01-31",
                anomaly_rate=1.5,
            )

        with pytest.raises(ValueError):
            TimeSeriesConfig(
                start_time="2024-01-01",
                end_time="2024-01-31",
                anomaly_rate=-0.1,
            )


class TestEventConfig:
    """Tests for EventConfig."""

    def test_basic_config(self) -> None:
        """Test basic event configuration."""
        config = EventConfig(
            event_types=["login", "logout"],
            count=10,
        )

        assert config.event_types == ["login", "logout"]
        assert config.count == 10
        assert config.include_metadata is True
        assert config.include_actor is True

    def test_full_config(self) -> None:
        """Test full event configuration."""
        config = EventConfig(
            event_types=["click", "scroll", "submit"],
            count=100,
            context="Web analytics",
            include_metadata=False,
            include_actor=False,
            time_range="last 7 days",
            severity_distribution={"low": 0.7, "medium": 0.2, "high": 0.1},
            correlation_groups=5,
        )

        assert config.context == "Web analytics"
        assert config.include_metadata is False
        assert config.correlation_groups == 5

    def test_count_validation(self) -> None:
        """Test count validation."""
        with pytest.raises(ValueError):
            EventConfig(
                event_types=["test"],
                count=0,
            )


class TestEmbeddingsConfig:
    """Tests for EmbeddingsConfig."""

    def test_basic_config(self) -> None:
        """Test basic embeddings configuration."""
        config = EmbeddingsConfig(
            content_type="documents",
            count=10,
        )

        assert config.content_type == "documents"
        assert config.count == 10
        assert config.include_embeddings is True

    def test_qa_pairs_config(self) -> None:
        """Test Q&A pairs configuration."""
        config = EmbeddingsConfig(
            content_type="qa_pairs",
            count=50,
            context="Customer support",
            categories=["billing", "technical"],
        )

        assert config.content_type == "qa_pairs"
        assert config.categories == ["billing", "technical"]

    def test_length_validation(self) -> None:
        """Test length validation."""
        with pytest.raises(ValueError):
            EmbeddingsConfig(
                content_type="documents",
                min_length=0,
            )

        with pytest.raises(ValueError):
            EmbeddingsConfig(
                content_type="documents",
                max_length=-1,
            )

    def test_content_types(self) -> None:
        """Test all valid content types."""
        valid_types = [
            "qa_pairs",
            "documents",
            "paragraphs",
            "sentences",
            "code_snippets",
            "product_descriptions",
            "reviews",
            "articles",
        ]

        for content_type in valid_types:
            config = EmbeddingsConfig(content_type=content_type)
            assert config.content_type == content_type
