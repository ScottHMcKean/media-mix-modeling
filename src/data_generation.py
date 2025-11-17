"""
Data generation module for MMM synthetic data.

This module provides functionality to generate synthetic media mix modeling data
with configurable channels, adstock effects, and saturation curves. Supports
saving to Databricks Delta tables.
"""

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from src.transforms import geometric_adstock, logistic_saturation


class ChannelConfig(BaseModel):
    """Configuration for a single media channel."""

    name: str
    beta: float = Field(description="Channel contribution coefficient")
    min_spend: float = Field(description="Minimum spend value")
    max_spend: float = Field(description="Maximum spend value")
    sigma: float = Field(default=1.0, description="Noise in spend signal")

    # Adstock configuration
    has_adstock: bool = Field(default=False)
    alpha: Optional[float] = Field(default=None, description="Decay rate for adstock")

    # Saturation configuration
    has_saturation: bool = Field(default=False)
    mu: Optional[float] = Field(default=None, description="Saturation parameter")


class DataGeneratorConfig(BaseModel):
    """Configuration for MMM data generator."""

    start_date: datetime
    end_date: datetime
    outcome_name: str = "sales"
    intercept: float = Field(description="Baseline outcome value")
    sigma: float = Field(description="Noise in outcome")
    scale: float = Field(description="Scale factor for outcome")
    channels: Dict[str, ChannelConfig] = Field(default_factory=dict)


class DataGenerator:
    """Generate synthetic MMM data with configurable channels and effects."""

    def __init__(self, config: DataGeneratorConfig):
        """
        Initialize data generator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.n_days = (config.end_date - config.start_date).days + 1

    @classmethod
    def from_yaml(cls, config_path: str) -> "DataGenerator":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            DataGenerator instance
        """
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Parse channel configurations
        channels = {}
        for name, channel_data in raw_config.get("media", {}).items():
            channels[name] = ChannelConfig(
                name=name,
                beta=channel_data["beta"],
                min_spend=channel_data["min"],
                max_spend=channel_data["max"],
                sigma=channel_data.get("signal", {}).get("sigma", 1.0),
                has_adstock=channel_data.get("decay", False),
                alpha=channel_data.get("alpha"),
                has_saturation=channel_data.get("saturation", False),
                mu=channel_data.get("mu"),
            )

        config = DataGeneratorConfig(
            start_date=raw_config["start_date"],
            end_date=raw_config["end_date"],
            outcome_name=raw_config["outcome"]["name"],
            intercept=raw_config["outcome"]["intercept"],
            sigma=raw_config["outcome"]["sigma"],
            scale=raw_config["outcome"]["scale"],
            channels=channels,
        )

        return cls(config)

    def _generate_spend(self, channel: ChannelConfig) -> np.ndarray:
        """
        Generate spend signal for a channel.

        Args:
            channel: Channel configuration

        Returns:
            Array of spend values
        """
        # Random walk with noise
        x = np.abs(np.cumsum(np.random.normal(0, channel.sigma, size=self.n_days)))

        # Rescale to min/max range
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)
        x = x * (channel.max_spend - channel.min_spend) + channel.min_spend

        return x

    def _compute_impact(self, spend: np.ndarray, channel: ChannelConfig) -> np.ndarray:
        """
        Compute impact of channel spend on outcome.

        Args:
            spend: Spend values
            channel: Channel configuration

        Returns:
            Impact values
        """
        # Normalize to 0-1 range for transformations
        spend_range = spend.max() - spend.min()
        if spend_range > 0:
            x = (spend - spend.min()) / spend_range
        else:
            # If all values are the same, use a constant normalized value
            x = np.ones_like(spend) * 0.5

        # Apply adstock if configured
        if channel.has_adstock and channel.alpha is not None:
            # Use numpy implementation for data generation
            weights = np.array([channel.alpha**i for i in range(12)])
            weights = weights / weights.sum()
            x = np.convolve(x, weights, mode="same")

        # Apply saturation if configured
        if channel.has_saturation and channel.mu is not None:
            # Use numpy implementation for data generation
            x = (1 - np.exp(-channel.mu * x)) / (1 + np.exp(-channel.mu * x))

        # Apply channel coefficient
        return channel.beta * x

    def generate(self) -> pd.DataFrame:
        """
        Generate synthetic MMM dataset.

        Returns:
            DataFrame with date index, channel spends, and outcome
        """
        # Create date range
        dates = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq="D")

        # Initialize dataframe
        data = {}

        # Baseline outcome with noise
        outcome = np.random.normal(self.config.intercept, self.config.sigma, self.n_days)

        # Add each channel's contribution
        for channel in self.config.channels.values():
            spend = self._generate_spend(channel)
            impact = self._compute_impact(spend, channel)

            data[channel.name] = np.round(spend, 2)
            outcome += impact

        # Scale and add outcome
        data[self.config.outcome_name] = np.round(outcome * self.config.scale, 2)

        # Create dataframe with date index
        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"

        return df

    def save_to_delta(
        self, df: pd.DataFrame, catalog: str, schema: str, table: str, mode: str = "overwrite"
    ):
        """
        Save generated data to Databricks Delta table.

        Args:
            df: DataFrame to save
            catalog: Unity Catalog name
            schema: Schema name
            table: Table name
            mode: Write mode (overwrite, append)
        """
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        # Convert pandas to Spark DataFrame
        df_reset = df.reset_index()
        sdf = spark.createDataFrame(df_reset)

        # Write to Delta table
        table_path = f"{catalog}.{schema}.{table}"
        sdf.write.format("delta").mode(mode).saveAsTable(table_path)

        print(f"Data saved to {table_path}")
