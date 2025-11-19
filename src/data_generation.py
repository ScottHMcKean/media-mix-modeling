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
from mlflow.models import ModelConfig

from src.data_io import load_data, save_data
from src.environment import has_spark
from src.transforms import geometric_adstock, logistic_saturation


class ChannelConfig(BaseModel):
    """Configuration for a single media channel."""

    name: str
    beta: float = Field(description="Channel contribution coefficient")
    min_spend: float = Field(description="Minimum spend value")
    max_spend: float = Field(description="Maximum spend value")
    sigma: float = Field(default=1.0, description="Noise in spend signal")
    has_adstock: bool = Field(default=False)
    alpha: Optional[float] = Field(default=None, description="Decay rate for adstock")
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
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    catalog: Optional[str] = Field(default="main", description="Unity Catalog name")
    schema_name: Optional[str] = Field(default="mmm", description="Schema name", alias="schema")
    synthetic_data_table: Optional[str] = Field(
        default="synthetic_mmm_data", description="Table name"
    )


class DataGenerator:
    """Generate synthetic MMM data with configurable channels and effects."""

    def __init__(self, config: DataGeneratorConfig):
        """
        Initialize data generator.

        Args:
            config: Configuration object
        """
        self.config = config
        # Calculate number of weeks instead of days
        self.n_periods = int((config.end_date - config.start_date).days / 7) + 1

    @classmethod
    def from_config(cls, raw_config: ModelConfig) -> "DataGenerator":
        """
        Load configuration from MLflow ModelConfig object.

        Args:
            config: MLflow ModelConfig object

        Returns:
            DataGenerator instance
        """
        # Get workspace and data sections
        workspace = raw_config.get("workspace")
        data_config = raw_config.get("data")

        # Parse channel configurations
        channels = {}
        media = data_config["media"]
        for name, channel_data in media.items():
            channels[name] = ChannelConfig(
                name=name,
                beta=channel_data["beta"],
                min_spend=channel_data["min"],
                max_spend=channel_data["max"],
                sigma=channel_data["sigma"],
                has_adstock=channel_data.get("decay", False),
                alpha=channel_data.get("alpha"),
                has_saturation=channel_data.get("saturation", False),
                mu=channel_data.get("mu"),
            )

        outcome = data_config["outcome"]

        gen_config = DataGeneratorConfig(
            start_date=data_config.get("start_date", "2020-01-01"),
            end_date=data_config.get("end_date", "2024-01-01"),
            outcome_name=outcome["name"],
            intercept=outcome["intercept"],
            sigma=outcome["sigma"],
            scale=outcome["scale"],
            channels=channels,
            random_seed=data_config.get("random_seed"),
            catalog=workspace["catalog"],
            schema_name=workspace["schema"],
            synthetic_data_table=data_config["table"],
        )

        return cls(gen_config)

    def _generate_spend(self, channel: ChannelConfig) -> np.ndarray:
        """
        Generate spend signal for a channel.

        Args:
            channel: Channel configuration

        Returns:
            Array of spend values
        """
        # Random walk with noise
        x = np.abs(np.cumsum(np.random.normal(0, channel.sigma, size=self.n_periods)))

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
            # Limit adstock window to the length of the data or 12, whichever is smaller
            adstock_window = min(len(x), 12)
            weights = np.array([channel.alpha**i for i in range(adstock_window)])
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
        # Set random seed if specified
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # Create date range with weekly frequency (Mondays)
        dates = pd.date_range(start=self.config.start_date, periods=self.n_periods, freq="W-MON")

        # Initialize dataframe
        data = {}

        # Baseline outcome with noise
        outcome = np.random.normal(self.config.intercept, self.config.sigma, self.n_periods)

        # Add each channel's contribution
        for channel in self.config.channels.values():
            spend = self._generate_spend(channel)
            impact = self._compute_impact(spend, channel)

            data[channel.name] = np.round(spend, 2)
            outcome += impact

        # Scale and add outcome
        data[self.config.outcome_name] = np.round(outcome * self.config.scale, 2)

        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"

        return df

    def save_to_delta(
        self,
        df: pd.DataFrame,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        mode: str = "overwrite",
    ):
        """
        Save generated data to Databricks Delta table.

        Uses values from config if not provided.

        Args:
            df: DataFrame to save
            catalog: Unity Catalog name (defaults to config value)
            schema: Schema name (defaults to config value)
            table: Table name (defaults to config value)
            mode: Write mode (overwrite, append)
        """
        # Use config values as defaults
        catalog = catalog or self.config.catalog
        schema = schema or self.config.schema_name
        table = table or self.config.synthetic_data_table

        # Use unified save function
        save_data(
            df=df,
            destination=table,
            catalog=catalog,
            schema=schema,
            table=table,
            use_delta=True,
            mode=mode,
        )

    def save(
        self,
        df: pd.DataFrame,
        file_path: Optional[str] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        mode: str = "overwrite",
        use_delta: Optional[bool] = None,
    ):
        """
        Save generated data to either local file or Delta table.

        Auto-detects environment if use_delta is None. If Spark is available and
        table info is provided, saves to Delta. Otherwise saves to local file.

        Args:
            df: DataFrame to save
            file_path: Local file path (for local save)
            catalog: Unity Catalog name (for Delta)
            schema: Schema name (for Delta)
            table: Table name (for Delta)
            mode: Write mode (overwrite, append)
            use_delta: Force Delta (True) or local (False), or auto-detect (None)
        """
        # Use config values as defaults
        catalog = catalog or self.config.catalog
        schema = schema or self.config.schema_name
        table = table or self.config.synthetic_data_table

        # Auto-detect if not specified
        if use_delta is None:
            use_delta = has_spark() and all([catalog, schema, table])

        if use_delta:
            self.save_to_delta(df, catalog=catalog, schema=schema, table=table, mode=mode)
        else:
            if not file_path:
                file_path = f"local_data/{table}.csv"
            save_data(df=df, destination=file_path, use_delta=False, mode=mode)

    @staticmethod
    def load(
        source: str,
        file_path: Optional[str] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        use_delta: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Load data from either local file or Delta table.

        Auto-detects environment if use_delta is None. If Spark is available and
        table info is provided, loads from Delta. Otherwise loads from local file.

        Args:
            source: Either a file path (for local) or table name (for Delta)
            file_path: Explicit local file path (optional)
            catalog: Unity Catalog name (for Delta)
            schema: Schema name (for Delta)
            table: Table name (for Delta)
            use_delta: Force Delta (True) or local (False), or auto-detect (None)

        Returns:
            DataFrame with data
        """
        return load_data(
            source=source,
            file_path=file_path,
            catalog=catalog,
            schema=schema,
            table=table,
            use_delta=use_delta,
        )
