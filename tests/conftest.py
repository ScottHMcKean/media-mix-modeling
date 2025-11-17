"""
Pytest configuration and fixtures for MMM tests.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data_generation import ChannelConfig, DataGenerator, DataGeneratorConfig
from src.model import ChannelSpec, MMModelConfig


@pytest.fixture
def sample_channel_config():
    """Sample channel configuration for testing."""
    return ChannelConfig(
        name="test_channel",
        beta=1.5,
        min_spend=1000,
        max_spend=10000,
        sigma=1.0,
        has_adstock=True,
        alpha=0.6,
        has_saturation=True,
        mu=2.0
    )


@pytest.fixture
def sample_data_generator_config(sample_channel_config):
    """Sample data generator configuration."""
    return DataGeneratorConfig(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 31),
        outcome_name="sales",
        intercept=5.0,
        sigma=0.5,
        scale=10000,
        channels={"test_channel": sample_channel_config}
    )


@pytest.fixture
def synthetic_data():
    """Generate synthetic MMM data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
    n = len(dates)
    
    data = {
        "channel_1": np.random.uniform(1000, 5000, n),
        "channel_2": np.random.uniform(2000, 8000, n),
        "sales": np.random.uniform(50000, 150000, n)
    }
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    
    return df


@pytest.fixture
def model_config():
    """Sample model configuration for testing."""
    channels = [
        ChannelSpec(
            name="channel_1",
            beta_prior_sigma=1.0,
            has_adstock=True,
            adstock_alpha_prior=3.0,
            adstock_beta_prior=2.0,
            has_saturation=False
        ),
        ChannelSpec(
            name="channel_2",
            beta_prior_sigma=1.0,
            has_adstock=False,
            has_saturation=True,
            saturation_k_prior_mean=0.5,
            saturation_s_prior_alpha=2.0,
            saturation_s_prior_beta=2.0
        ),
    ]
    
    return MMModelConfig(
        outcome_name="sales",
        intercept_mu=0.0,
        intercept_sigma=5.0,
        sigma_prior_beta=1.0,
        outcome_scale=10000,
        channels=channels
    )


@pytest.fixture
def small_synthetic_data():
    """Small synthetic dataset for faster tests."""
    np.random.seed(42)
    
    dates = pd.date_range(start="2022-01-01", periods=50, freq="D")
    n = len(dates)
    
    data = {
        "channel_1": np.random.uniform(1000, 5000, n),
        "channel_2": np.random.uniform(2000, 8000, n),
        "sales": np.random.uniform(50000, 150000, n)
    }
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    
    return df

