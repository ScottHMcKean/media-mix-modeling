"""
Tests for performance analysis functionality.
"""

import pandas as pd
import pytest

from src.model import ChannelSpec, MMMModelConfig, MediaMixModel


@pytest.fixture
def sample_data():
    """Create sample data with date range."""
    dates = pd.date_range(start="2020-01-01", periods=104, freq="W-MON")  # 2 years
    data = {
        "channel1": [100 + i * 10 for i in range(104)],
        "channel2": [50 + i * 5 for i in range(104)],
        "sales": [1000 + i * 50 for i in range(104)],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def simple_model_config():
    """Create a simple model configuration."""
    channels = [
        ChannelSpec(
            name="channel1",
            beta_prior_sigma=1.0,
            has_adstock=False,
            has_saturation=False,
        ),
        ChannelSpec(
            name="channel2",
            beta_prior_sigma=1.0,
            has_adstock=False,
            has_saturation=False,
        ),
    ]

    return MMMModelConfig(
        outcome_name="sales",
        outcome_scale=1000.0,
        channels=channels,
        include_trend=False,
    )


@pytest.fixture
def fitted_model(simple_model_config, sample_data):
    """Create a fitted model."""
    model = MediaMixModel(simple_model_config)
    # Fit with minimal samples for speed
    model.fit(sample_data, draws=100, tune=100, chains=2)
    return model


class TestPerformanceAnalysis:
    """Test performance analysis methods."""

    def test_get_performance_analysis_full_period(self, fitted_model, sample_data):
        """Test performance analysis for full period."""
        analysis = fitted_model.get_performance_analysis(
            df=sample_data, analysis_name="full_period"
        )

        # Check structure
        assert isinstance(analysis, pd.DataFrame)
        assert "analysis_name" in analysis.columns
        assert "start_date" in analysis.columns
        assert "end_date" in analysis.columns
        assert "channel" in analysis.columns
        assert "total_contribution" in analysis.columns
        assert "total_spend" in analysis.columns
        assert "roas" in analysis.columns
        assert "pct_of_total_sales" in analysis.columns
        assert "pct_of_incremental_sales" in analysis.columns

        # Check metadata
        assert all(analysis["analysis_name"] == "full_period")
        assert analysis["start_date"].iloc[0] == "2020-01-06"
        assert analysis["end_date"].iloc[0] == "2021-12-27"

        # Should have rows for both channels plus base
        assert len(analysis) == 3  # channel1, channel2, base

    def test_get_performance_analysis_with_date_range(self, fitted_model, sample_data):
        """Test performance analysis with specific date range."""
        analysis = fitted_model.get_performance_analysis(
            df=sample_data,
            start_date="2020-06-01",
            end_date="2020-12-31",
            analysis_name="h2_2020",
        )

        # Check metadata
        assert all(analysis["analysis_name"] == "h2_2020")
        assert analysis["start_date"].iloc[0] == "2020-06-01"
        assert analysis["end_date"].iloc[0] == "2020-12-31"

        # Should have data for filtered period
        assert len(analysis) == 3  # channel1, channel2, base

    def test_get_performance_analysis_start_date_only(self, fitted_model, sample_data):
        """Test performance analysis with only start date."""
        analysis = fitted_model.get_performance_analysis(
            df=sample_data, start_date="2021-01-01", analysis_name="2021_onwards"
        )

        # Start date should be specified
        assert analysis["start_date"].iloc[0] == "2021-01-01"
        # End date should be last date in data
        assert analysis["end_date"].iloc[0] == "2021-12-27"

    def test_get_performance_analysis_end_date_only(self, fitted_model, sample_data):
        """Test performance analysis with only end date."""
        analysis = fitted_model.get_performance_analysis(
            df=sample_data, end_date="2020-12-31", analysis_name="2020"
        )

        # Start date should be first date in data
        assert analysis["start_date"].iloc[0] == "2020-01-06"
        # End date should be specified
        assert analysis["end_date"].iloc[0] == "2020-12-31"

    def test_get_performance_analysis_invalid_date_range(self, fitted_model, sample_data):
        """Test performance analysis with invalid date range."""
        with pytest.raises(ValueError, match="No data found"):
            fitted_model.get_performance_analysis(
                df=sample_data, start_date="2025-01-01", end_date="2025-12-31"
            )

    def test_get_performance_analysis_channels(self, fitted_model, sample_data):
        """Test that performance analysis includes all channels."""
        analysis = fitted_model.get_performance_analysis(df=sample_data)

        channels = analysis["channel"].tolist()
        assert "channel1" in channels
        assert "channel2" in channels
        assert "base" in channels

    def test_get_performance_analysis_metrics_types(self, fitted_model, sample_data):
        """Test that metrics have correct types."""
        analysis = fitted_model.get_performance_analysis(df=sample_data)

        # Numeric columns
        assert pd.api.types.is_numeric_dtype(analysis["total_contribution"])
        assert pd.api.types.is_numeric_dtype(analysis["total_spend"])
        assert pd.api.types.is_numeric_dtype(analysis["roas"])
        assert pd.api.types.is_numeric_dtype(analysis["pct_of_total_sales"])
        assert pd.api.types.is_numeric_dtype(analysis["pct_of_incremental_sales"])

    def test_performance_analysis_comparison(self, fitted_model, sample_data):
        """Test comparing performance across different periods."""
        # Analyze first half
        analysis_h1 = fitted_model.get_performance_analysis(
            df=sample_data,
            start_date="2020-01-01",
            end_date="2020-06-30",
            analysis_name="h1_2020",
        )

        # Analyze second half
        analysis_h2 = fitted_model.get_performance_analysis(
            df=sample_data,
            start_date="2020-07-01",
            end_date="2020-12-31",
            analysis_name="h2_2020",
        )

        # Should have same structure
        assert list(analysis_h1.columns) == list(analysis_h2.columns)
        assert len(analysis_h1) == len(analysis_h2)

        # Should have different analysis names
        assert analysis_h1["analysis_name"].iloc[0] == "h1_2020"
        assert analysis_h2["analysis_name"].iloc[0] == "h2_2020"

        # Metrics should be different (due to different data)
        channel1_h1 = analysis_h1[analysis_h1["channel"] == "channel1"]
        channel1_h2 = analysis_h2[analysis_h2["channel"] == "channel1"]

        # Total spend should be different (different number of weeks)
        assert channel1_h1["total_spend"].iloc[0] != channel1_h2["total_spend"].iloc[0]


class TestFromConfig:
    """Test from_config class method."""

    def test_from_config_creates_model(self):
        """Test creating model from config dict."""
        config_dict = {
            "outcome_name": "sales",
            "outcome_scale": 1000.0,
            "include_trend": True,
            "trend_prior_sigma": 0.5,
            "priors": {
                "intercept_mu": 5.0,
                "intercept_sigma": 2.0,
                "sigma_beta": 1.0,
            },
            "channels": {
                "channel1": {
                    "beta_prior_sigma": 2.0,
                    "has_adstock": True,
                    "adstock_alpha_prior": 4.5,
                    "adstock_beta_prior": 2.5,
                    "has_saturation": True,
                    "saturation_k_prior_mean": 0.5,
                    "saturation_s_prior_alpha": 3.0,
                    "saturation_s_prior_beta": 3.0,
                },
                "channel2": {
                    "beta_prior_sigma": 1.0,
                    "has_adstock": False,
                    "has_saturation": False,
                },
            },
        }

        mmm = MediaMixModel.from_config(config_dict)

        # Check model configuration
        assert mmm.config.outcome_name == "sales"
        assert mmm.config.outcome_scale == 1000.0
        assert mmm.config.include_trend is True
        assert mmm.config.trend_prior_sigma == 0.5
        assert len(mmm.config.channels) == 2

        # Check channel configurations
        channel1 = next(c for c in mmm.config.channels if c.name == "channel1")
        assert channel1.has_adstock is True
        assert channel1.has_saturation is True
        assert channel1.beta_prior_sigma == 2.0

        channel2 = next(c for c in mmm.config.channels if c.name == "channel2")
        assert channel2.has_adstock is False
        assert channel2.has_saturation is False

    def test_from_config_with_mlflow_config(self):
        """Test creating model from MLflow ModelConfig."""
        import mlflow
        import tempfile
        from pathlib import Path

        # Create temporary config file
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            config_content = """
model:
  outcome_name: sales
  outcome_scale: 100000
  include_trend: true
  trend_prior_sigma: 0.5
  priors:
    intercept_mu: 5.0
    intercept_sigma: 2.0
    sigma_beta: 1.0
  channels:
    adwords:
      beta_prior_sigma: 2.0
      has_adstock: true
      adstock_alpha_prior: 4.5
      adstock_beta_prior: 2.5
      has_saturation: true
      saturation_k_prior_mean: 0.5
      saturation_s_prior_alpha: 3.0
      saturation_s_prior_beta: 3.0
"""
            config_path.write_text(config_content)

            # Load using MLflow ModelConfig
            config = mlflow.models.ModelConfig(development_config=str(config_path))
            model_config = config.get("model")

            # Create model from config
            mmm = MediaMixModel.from_config(model_config)

            # Verify
            assert mmm.config.outcome_name == "sales"
            assert mmm.config.outcome_scale == 100000
            assert len(mmm.config.channels) == 1
            assert mmm.config.channels[0].name == "adwords"


class TestMLflowConfig:
    """Test MLflow configuration integration."""

    def test_save_to_mlflow_with_config(self, fitted_model):
        """Test saving to MLflow using config dict."""
        import mlflow
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")

            mlflow_config = {
                "experiment_name": "test_experiment",
                "run_name": "test_run",
                "model_name": "test_model",
                "register_model": False,
            }

            run_id = fitted_model.save_to_mlflow(mlflow_config=mlflow_config)

            assert run_id is not None
            run = mlflow.get_run(run_id)
            assert run.info.run_name == "test_run"

    def test_save_to_mlflow_config_override(self, fitted_model):
        """Test that explicit parameters override config."""
        import mlflow
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")

            mlflow_config = {
                "experiment_name": "config_experiment",
                "run_name": "config_run",
            }

            # Override with explicit parameter
            run_id = fitted_model.save_to_mlflow(
                mlflow_config=mlflow_config, run_name="override_run"
            )

            run = mlflow.get_run(run_id)
            # Should use overridden value
            assert run.info.run_name == "override_run"

    def test_save_to_mlflow_no_experiment_name_raises_error(self, fitted_model):
        """Test that missing experiment_name raises error."""
        with pytest.raises(ValueError, match="experiment_name is required"):
            fitted_model.save_to_mlflow()

    def test_save_to_mlflow_register_without_model_name_raises_error(self, fitted_model):
        """Test that register_model=True without model_name raises error."""
        import mlflow
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")

            with pytest.raises(ValueError, match="model_name is required"):
                fitted_model.save_to_mlflow(
                    experiment_name="test", register_model=True, model_name=None
                )
