"""
Tests for model module.
"""

import pytest
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from src.model import ChannelSpec, MediaMixModel, MMMModelConfig


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

    # Check that log likelihood is computed (required for WAIC/LOO)
    assert "log_likelihood" in idata.groups()
    assert "sales" in idata.log_likelihood.data_vars


def test_waic_loo_computation(model_config, small_synthetic_data):
    """Test that WAIC and LOO can be computed after fitting."""
    model = MediaMixModel(model_config)

    # Fit with minimal sampling for testing
    idata = model.fit(df=small_synthetic_data, draws=50, tune=50, chains=1, target_accept=0.8)

    # Test WAIC computation - should not raise an error
    waic = az.waic(idata)
    assert waic is not None
    # WAIC result has elpd_waic attribute
    assert hasattr(waic, "elpd_waic")
    assert np.isfinite(waic.elpd_waic)

    # Test LOO computation - should not raise an error
    loo = az.loo(idata)
    assert loo is not None
    # LOO result has elpd_loo attribute
    assert hasattr(loo, "elpd_loo")
    assert np.isfinite(loo.elpd_loo)


def test_model_config_with_minimal_channels():
    """Test model with minimal channel configuration."""
    channel = ChannelSpec(name="simple_channel", beta_prior_sigma=1.0)

    config = MMMModelConfig(outcome_name="outcome", channels=[channel])

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

    config = MMMModelConfig(outcome_name="sales", outcome_scale=10000, channels=[channel])

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

    config = MMMModelConfig(outcome_name="sales", outcome_scale=10000, channels=[channel])

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

    config = MMMModelConfig(
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


def test_channel_contributions_include_base(model_config, small_synthetic_data):
    """Test that channel contributions include base sales."""
    model = MediaMixModel(model_config)

    # Fit with minimal sampling for testing
    model.fit(df=small_synthetic_data, draws=50, tune=50, chains=1, target_accept=0.8)

    # Get contributions
    contributions = model.get_channel_contributions(small_synthetic_data)

    # Check that base is included
    assert "base" in contributions.columns
    assert contributions["base"].sum() > 0

    # Check all channels are present
    assert "channel_1" in contributions.columns
    assert "channel_2" in contributions.columns


def test_channel_performance_summary(model_config, small_synthetic_data):
    """Test channel performance summary with ROAS and percentages."""
    model = MediaMixModel(model_config)

    # Fit with minimal sampling for testing
    model.fit(df=small_synthetic_data, draws=50, tune=50, chains=1, target_accept=0.8)

    # Get performance summary
    performance = model.get_channel_performance_summary(small_synthetic_data)

    # Check structure
    assert isinstance(performance, pd.DataFrame)
    assert "channel" in performance.columns
    assert "total_contribution" in performance.columns
    assert "total_spend" in performance.columns
    assert "roas" in performance.columns
    assert "pct_of_total_sales" in performance.columns
    assert "pct_of_incremental_sales" in performance.columns

    # Check that base is included
    assert "base" in performance["channel"].values

    # Check all channels are present
    assert "channel_1" in performance["channel"].values
    assert "channel_2" in performance["channel"].values

    # Check ROAS is positive for channels
    channel_rows = performance[performance["channel"] != "base"]
    assert (channel_rows["roas"] >= 0).all()

    # Check percentages sum to approximately 100% (allow for model attribution differences)
    total_pct = performance["pct_of_total_sales"].sum()
    assert 95 < total_pct < 105  # Allow for attribution differences
