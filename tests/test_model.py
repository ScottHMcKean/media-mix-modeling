"""
Tests for model module.
"""

import pytest
import numpy as np
import pandas as pd
import pymc as pm

from src.model import ChannelSpec, MediaMixModel, MMModelConfig


def test_channel_spec_creation():
    """Test creating a channel specification."""
    spec = ChannelSpec(
        name="test_channel",
        beta_prior_sigma=2.0,
        has_adstock=True,
        adstock_alpha_prior=5.0,
        adstock_beta_prior=2.0,
        has_saturation=True,
        saturation_k_prior_mean=0.5,
        saturation_s_prior_alpha=3.0,
        saturation_s_prior_beta=3.0,
    )

    assert spec.name == "test_channel"
    assert spec.has_adstock is True
    assert spec.has_saturation is True


def test_model_config_creation(model_config):
    """Test creating model configuration."""
    assert model_config.outcome_name == "sales"
    assert len(model_config.channels) == 2
    assert model_config.outcome_scale == 10000


def test_model_initialization(model_config):
    """Test initializing a model."""
    model = MediaMixModel(model_config)

    assert model.config == model_config
    assert model.model is None
    assert model.idata is None
    assert len(model.scalers) == 0


def test_scale_data(model_config, synthetic_data):
    """Test data scaling."""
    model = MediaMixModel(model_config)

    df_scaled = model._scale_data(synthetic_data)

    # Check that channels are scaled to 0-1
    for channel in ["channel_1", "channel_2"]:
        assert df_scaled[channel].min() >= 0
        assert df_scaled[channel].max() <= 1

    # Check that scalers are created
    assert len(model.scalers) == 2
    assert "channel_1" in model.scalers
    assert "channel_2" in model.scalers

    # Check outcome is scaled
    assert df_scaled["sales"].mean() < synthetic_data["sales"].mean()


def test_build_model(model_config, small_synthetic_data):
    """Test building PyMC model."""
    model = MediaMixModel(model_config)

    df_scaled = model._scale_data(small_synthetic_data)
    pymc_model = model.build_model(df_scaled)

    assert isinstance(pymc_model, pm.Model)
    assert model.model is not None

    # Check that model has expected variables
    var_names = [v.name for v in pymc_model.unobserved_value_vars]
    assert "intercept" in var_names
    assert "beta_channel_1" in var_names
    assert "beta_channel_2" in var_names
    assert "sigma" in var_names


def test_fit_model(model_config, small_synthetic_data):
    """Test fitting model with MCMC (slow test)."""
    model = MediaMixModel(model_config)

    # Fit with minimal sampling for testing
    idata = model.fit(df=small_synthetic_data, draws=50, tune=50, chains=1, target_accept=0.8)

    assert idata is not None
    assert model.idata is not None

    # Check that posterior samples exist
    assert "posterior" in idata.groups()
    assert "intercept" in idata.posterior.data_vars


def test_model_config_with_minimal_channels():
    """Test model with minimal channel configuration."""
    channel = ChannelSpec(name="simple_channel", beta_prior_sigma=1.0)

    config = MMModelConfig(outcome_name="outcome", channels=[channel])

    assert len(config.channels) == 1
    assert config.channels[0].has_adstock is False
    assert config.channels[0].has_saturation is False


def test_build_model_with_adstock_only():
    """Test building model with only adstock transformation."""
    channel = ChannelSpec(
        name="channel_1",
        beta_prior_sigma=1.0,
        has_adstock=True,
        adstock_alpha_prior=3.0,
        adstock_beta_prior=2.0,
        has_saturation=False,
    )

    config = MMModelConfig(outcome_name="sales", outcome_scale=10000, channels=[channel])

    # Create simple data
    df = pd.DataFrame(
        {
            "channel_1": np.random.uniform(1000, 5000, 30),
            "sales": np.random.uniform(50000, 100000, 30),
        }
    )

    model = MediaMixModel(config)
    df_scaled = model._scale_data(df)
    pymc_model = model.build_model(df_scaled)

    var_names = [v.name for v in pymc_model.unobserved_value_vars]
    assert "adstock_alpha_channel_1" in var_names
    assert "saturation_k_channel_1" not in var_names


def test_build_model_with_saturation_only():
    """Test building model with only saturation transformation."""
    channel = ChannelSpec(
        name="channel_1",
        beta_prior_sigma=1.0,
        has_adstock=False,
        has_saturation=True,
        saturation_k_prior_mean=0.5,
        saturation_s_prior_alpha=2.0,
        saturation_s_prior_beta=2.0,
    )

    config = MMModelConfig(outcome_name="sales", outcome_scale=10000, channels=[channel])

    # Create simple data
    df = pd.DataFrame(
        {
            "channel_1": np.random.uniform(1000, 5000, 30),
            "sales": np.random.uniform(50000, 100000, 30),
        }
    )

    model = MediaMixModel(config)
    df_scaled = model._scale_data(df)
    pymc_model = model.build_model(df_scaled)

    var_names = [v.name for v in pymc_model.unobserved_value_vars]
    assert "saturation_k_channel_1" in var_names
    assert "saturation_s_channel_1" in var_names
    assert "adstock_alpha_channel_1" not in var_names


def test_model_config_dict():
    """Test converting model config to dictionary."""
    channel = ChannelSpec(
        name="test",
        beta_prior_sigma=2.0,
        has_adstock=True,
        adstock_alpha_prior=5.0,
        adstock_beta_prior=2.0,
    )

    config = MMModelConfig(
        outcome_name="sales",
        intercept_mu=1.0,
        intercept_sigma=2.0,
        sigma_prior_beta=1.5,
        outcome_scale=10000,
        channels=[channel],
    )

    config_dict = config.model_dump()

    assert config_dict["outcome_name"] == "sales"
    assert config_dict["intercept_mu"] == 1.0
    assert len(config_dict["channels"]) == 1
    assert config_dict["channels"][0]["name"] == "test"
