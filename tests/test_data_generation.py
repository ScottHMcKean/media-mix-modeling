"""
Tests for data generation module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.data_generation import (
    ChannelConfig,
    DataGenerator,
    DataGeneratorConfig
)


def test_channel_config_creation(sample_channel_config):
    """Test creating a channel configuration."""
    assert sample_channel_config.name == "test_channel"
    assert sample_channel_config.beta == 1.5
    assert sample_channel_config.has_adstock is True
    assert sample_channel_config.has_saturation is True


def test_data_generator_config_creation(sample_data_generator_config):
    """Test creating a data generator configuration."""
    config = sample_data_generator_config
    
    assert config.outcome_name == "sales"
    assert config.intercept == 5.0
    assert len(config.channels) == 1
    assert "test_channel" in config.channels


def test_data_generator_initialization(sample_data_generator_config):
    """Test initializing a data generator."""
    generator = DataGenerator(sample_data_generator_config)
    
    assert generator.config == sample_data_generator_config
    assert generator.n_days == 31  # Jan 1 to Jan 31


def test_generate_spend(sample_data_generator_config):
    """Test generating spend signal."""
    np.random.seed(42)
    
    generator = DataGenerator(sample_data_generator_config)
    channel = sample_data_generator_config.channels["test_channel"]
    
    spend = generator._generate_spend(channel)
    
    assert len(spend) == generator.n_days
    assert spend.min() >= channel.min_spend
    assert spend.max() <= channel.max_spend


def test_compute_impact(sample_data_generator_config):
    """Test computing channel impact."""
    np.random.seed(42)
    
    generator = DataGenerator(sample_data_generator_config)
    channel = sample_data_generator_config.channels["test_channel"]
    
    spend = np.random.uniform(1000, 10000, 31)
    impact = generator._compute_impact(spend, channel)
    
    assert len(impact) == len(spend)
    assert isinstance(impact, np.ndarray)


def test_generate_data(sample_data_generator_config):
    """Test generating complete dataset."""
    np.random.seed(42)
    
    generator = DataGenerator(sample_data_generator_config)
    df = generator.generate()
    
    # Check structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 31
    assert df.index.name == "date"
    
    # Check columns
    assert "test_channel" in df.columns
    assert "sales" in df.columns
    
    # Check data types
    assert df["test_channel"].dtype == np.float64
    assert df["sales"].dtype == np.float64


def test_generate_data_multiple_channels():
    """Test generating data with multiple channels."""
    np.random.seed(42)
    
    channels = {
        "tv": ChannelConfig(
            name="tv",
            beta=2.0,
            min_spend=5000,
            max_spend=20000,
            sigma=1.0,
            has_adstock=True,
            alpha=0.7,
            has_saturation=True,
            mu=3.0
        ),
        "digital": ChannelConfig(
            name="digital",
            beta=1.5,
            min_spend=2000,
            max_spend=15000,
            sigma=1.2,
            has_adstock=False,
            has_saturation=True,
            mu=2.5
        ),
    }
    
    config = DataGeneratorConfig(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 31),
        outcome_name="revenue",
        intercept=10.0,
        sigma=1.0,
        scale=50000,
        channels=channels
    )
    
    generator = DataGenerator(config)
    df = generator.generate()
    
    assert len(df) == 31
    assert "tv" in df.columns
    assert "digital" in df.columns
    assert "revenue" in df.columns


def test_data_generator_from_config(tmp_path):
    """Test loading generator from MLflow config."""
    import mlflow
    
    yaml_content = """
random_seed: 42
start_date: 2022-01-01
end_date: 2022-01-31
outcome:
  name: sales
  intercept: 5.0
  sigma: 0.5
  scale: 10000
media:
  test_channel:
    beta: 1.5
    min: 1000
    max: 10000
    sigma: 1.0
    decay: true
    alpha: 0.6
    saturation: true
    mu: 2.0
"""
    
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)
    
    config = mlflow.models.ModelConfig(development_config=str(yaml_file))
    generator = DataGenerator.from_config(config)
    
    assert generator.config.outcome_name == "sales"
    assert len(generator.config.channels) == 1
    assert "test_channel" in generator.config.channels
    assert generator.config.random_seed == 42
    
    # Generate data to ensure it works
    df = generator.generate()
    assert len(df) == 31


def test_channel_impact_with_adstock():
    """Test that adstock affects the impact calculation."""
    np.random.seed(42)
    
    # Channel with adstock
    channel_adstock = ChannelConfig(
        name="ch1",
        beta=1.0,
        min_spend=1000,
        max_spend=5000,
        sigma=1.0,
        has_adstock=True,
        alpha=0.8,
        has_saturation=False
    )
    
    # Channel without adstock
    channel_no_adstock = ChannelConfig(
        name="ch2",
        beta=1.0,
        min_spend=1000,
        max_spend=5000,
        sigma=1.0,
        has_adstock=False,
        has_saturation=False
    )
    
    config = DataGeneratorConfig(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 31),
        outcome_name="sales",
        intercept=5.0,
        sigma=0.1,
        scale=1000,
        channels={"ch1": channel_adstock}
    )
    
    generator = DataGenerator(config)
    
    # Create impulse spend (single spike)
    spend = np.zeros(31)
    spend[15] = 5000
    
    impact_adstock = generator._compute_impact(spend, channel_adstock)
    impact_no_adstock = generator._compute_impact(spend, channel_no_adstock)
    
    # With adstock, impact should be spread over time
    # Without adstock, impact should be concentrated
    assert impact_adstock[15] < impact_adstock[15:20].sum()
    assert impact_no_adstock[15] >= impact_no_adstock[16]


def test_channel_impact_with_saturation():
    """Test that saturation affects the impact calculation."""
    np.random.seed(42)
    
    channel = ChannelConfig(
        name="ch",
        beta=1.0,
        min_spend=0,
        max_spend=10000,
        sigma=1.0,
        has_adstock=False,
        has_saturation=True,
        mu=2.0
    )
    
    config = DataGeneratorConfig(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 10),
        outcome_name="sales",
        intercept=5.0,
        sigma=0.1,
        scale=1000,
        channels={"ch": channel}
    )
    
    generator = DataGenerator(config)
    
    # Low spend
    low_spend = np.ones(10) * 1000
    impact_low = generator._compute_impact(low_spend, channel)
    
    # High spend
    high_spend = np.ones(10) * 9000
    impact_high = generator._compute_impact(high_spend, channel)
    
    # With saturation, doubling spend should not double impact
    assert impact_high.mean() < impact_low.mean() * 9

