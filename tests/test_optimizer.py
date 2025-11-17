"""
Tests for optimizer module.
"""

import pytest
import numpy as np

from src.model import MMModelConfig
from src.optimizer import BudgetConstraints, BudgetOptimizer, OptimizationResult


def test_budget_constraints_creation():
    """Test creating budget constraints."""
    constraints = BudgetConstraints(
        total_budget=100000,
        min_spend_per_channel={
            "channel_1": 10000,
            "channel_2": 5000
        },
        max_spend_per_channel={
            "channel_1": 60000,
            "channel_2": 50000
        }
    )
    
    assert constraints.total_budget == 100000
    assert len(constraints.min_spend_per_channel) == 2
    assert len(constraints.max_spend_per_channel) == 2


def test_optimization_result_creation():
    """Test creating optimization result."""
    result = OptimizationResult(
        optimal_allocation={"channel_1": 60000, "channel_2": 40000},
        expected_outcome=500000,
        channel_roas={"channel_1": 5.0, "channel_2": 4.0},
        recommendations=["Increase channel_1 spend"]
    )
    
    assert result.optimal_allocation["channel_1"] == 60000
    assert result.expected_outcome == 500000
    assert len(result.recommendations) == 1


def test_optimizer_initialization(model_config):
    """Test initializing optimizer."""
    from src.model import MediaMixModel
    
    model = MediaMixModel(model_config)
    optimizer = BudgetOptimizer(
        model=model,
        channel_names=["channel_1", "channel_2"]
    )
    
    assert optimizer.model == model
    assert optimizer.channel_names == ["channel_1", "channel_2"]


def test_objective_function(model_config, small_synthetic_data):
    """Test objective function calculation."""
    from src.model import MediaMixModel
    
    model = MediaMixModel(model_config)
    model._scale_data(small_synthetic_data)
    
    optimizer = BudgetOptimizer(
        model=model,
        channel_names=["channel_1", "channel_2"]
    )
    
    # Mock posterior parameters
    posterior_params = {
        "intercept": 1.0,
        "beta_channel_1": 0.5,
        "beta_channel_2": 0.3,
        "adstock_alpha_channel_1": 0.6
    }
    
    allocation = np.array([50000, 30000])
    
    # Should return a scalar value
    result = optimizer._objective_function(allocation, posterior_params)
    assert isinstance(result, (float, np.floating))


def test_generate_recommendations():
    """Test generating recommendations."""
    from src.model import MediaMixModel
    
    model = MediaMixModel(MMModelConfig(
        outcome_name="sales",
        outcome_scale=10000,
        channels=[]
    ))
    
    optimizer = BudgetOptimizer(
        model=model,
        channel_names=["channel_1", "channel_2"]
    )
    
    allocation = {"channel_1": 60000, "channel_2": 40000}
    roas = {"channel_1": 5.0, "channel_2": 2.0}
    constraints = BudgetConstraints(
        total_budget=100000,
        min_spend_per_channel={"channel_1": 10000, "channel_2": 5000},
        max_spend_per_channel={"channel_1": 60000, "channel_2": 50000}
    )
    
    recommendations = optimizer._generate_recommendations(
        allocation, roas, constraints
    )
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert any("channel_1" in rec for rec in recommendations)


def test_constraints_validation():
    """Test that budget constraints are valid."""
    # Valid constraints
    constraints = BudgetConstraints(
        total_budget=100000,
        min_spend_per_channel={"ch1": 10000, "ch2": 20000},
        max_spend_per_channel={"ch1": 50000, "ch2": 60000}
    )
    
    assert constraints.total_budget == 100000
    
    # Check that max >= min for all channels
    for channel in constraints.min_spend_per_channel:
        assert (
            constraints.max_spend_per_channel[channel] >= 
            constraints.min_spend_per_channel[channel]
        )

