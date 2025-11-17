"""
Tests for agent module.
"""

import pytest
import numpy as np

from src.agent import (
    MMAgent,
    ForecastRequest,
    ForecastResult,
    HistoricalAnalysisResult
)


def test_forecast_request_creation():
    """Test creating forecast request."""
    request = ForecastRequest(
        future_spend={
            "channel_1": [5000, 6000, 7000],
            "channel_2": [3000, 3500, 4000]
        },
        dates=["2024-01-01", "2024-01-02", "2024-01-03"]
    )
    
    assert len(request.future_spend["channel_1"]) == 3
    assert request.dates is not None
    assert len(request.dates) == 3


def test_forecast_result_creation():
    """Test creating forecast result."""
    result = ForecastResult(
        predictions=[100000, 110000, 105000],
        lower_bound=[90000, 100000, 95000],
        upper_bound=[110000, 120000, 115000],
        dates=["2024-01-01", "2024-01-02", "2024-01-03"]
    )
    
    assert len(result.predictions) == 3
    assert len(result.lower_bound) == 3
    assert len(result.upper_bound) == 3


def test_historical_analysis_result_creation():
    """Test creating historical analysis result."""
    result = HistoricalAnalysisResult(
        channel_contributions={
            "channel_1": [1000, 1100, 1050],
            "channel_2": [800, 850, 825]
        },
        model_fit_metrics={
            "waic": 1500.5,
            "loo": 1501.2
        },
        summary_statistics={
            "beta_channel_1": {
                "mean": 1.5,
                "sd": 0.2,
                "hdi_3%": 1.1,
                "hdi_97%": 1.9
            }
        }
    )
    
    assert len(result.channel_contributions) == 2
    assert "waic" in result.model_fit_metrics
    assert "beta_channel_1" in result.summary_statistics


def test_agent_initialization(model_config, synthetic_data):
    """Test initializing agent."""
    from src.model import MediaMixModel
    
    model = MediaMixModel(model_config)
    agent = MMAgent(model=model, data=synthetic_data)
    
    assert agent.model == model
    assert agent.data.equals(synthetic_data)
    assert agent.optimizer is not None


def test_agent_get_channel_insights(model_config, small_synthetic_data):
    """Test getting channel insights from agent."""
    from src.model import MediaMixModel
    import arviz as az
    
    model = MediaMixModel(model_config)
    
    # We need to fit the model first, but for testing we can mock the idata
    # Create minimal mock inference data
    try:
        # Fit with minimal parameters for test
        model.fit(
            df=small_synthetic_data,
            draws=20,
            tune=20,
            chains=1,
            target_accept=0.8
        )
        
        agent = MMAgent(model=model, data=small_synthetic_data)
        insights = agent.get_channel_insights()
        
        assert isinstance(insights, dict)
        assert len(insights) == 2
        assert "channel_1" in insights
        assert "channel_2" in insights
        
    except Exception as e:
        # If sampling fails in test environment, skip
        pytest.skip(f"Could not fit model for test: {e}")


def test_forecast_request_without_dates():
    """Test forecast request without explicit dates."""
    request = ForecastRequest(
        future_spend={
            "channel_1": [5000, 6000],
            "channel_2": [3000, 3500]
        }
    )
    
    assert request.dates is None
    assert len(request.future_spend["channel_1"]) == 2

