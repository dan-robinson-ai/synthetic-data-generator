"""Time series generation example."""

from agentic_synth import GeneratorConfig, SyntheticGenerator
from agentic_synth.generators import TimeSeriesConfig


def main() -> None:
    """Demonstrate time series data generation."""

    config = GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini",
    )

    with SyntheticGenerator(config) as generator:
        # Generate stock price data
        print("Generating stock price data...")
        stock_config = TimeSeriesConfig(
            start_time="2024-01-01",
            end_time="2024-01-07",
            frequency="1d",
            columns=["open", "high", "low", "close", "volume"],
            context="Stock market trading data for a tech company (ACME Inc.)",
            trend="up",
            volatility="medium",
        )

        stock_data = generator.generate_time_series(stock_config)

        print("\nStock Price Data:")
        for point in stock_data[:5]:  # Show first 5 points
            print(f"  {point.get('timestamp')}: Close=${point.get('close', 'N/A')}")

        # Generate IoT sensor data
        print("\nGenerating IoT sensor data...")
        iot_config = TimeSeriesConfig(
            start_time="2024-06-01T00:00:00",
            end_time="2024-06-01T01:00:00",
            frequency="5m",
            columns=["temperature", "humidity", "pressure"],
            context="Industrial IoT sensors in a manufacturing facility",
            trend="cyclical",
            volatility="low",
            include_anomalies=True,
            anomaly_rate=0.1,
        )

        iot_data = generator.generate_time_series(iot_config)

        print("\nIoT Sensor Data:")
        for point in iot_data[:5]:
            print(
                f"  {point.get('timestamp')}: "
                f"Temp={point.get('temperature', 'N/A')}°C, "
                f"Humidity={point.get('humidity', 'N/A')}%"
            )


if __name__ == "__main__":
    main()
