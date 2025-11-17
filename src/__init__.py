"""
Media Mix Modeling - Simplified and Modular Implementation.

This package provides:
1. Data generation with synthetic MMM data
2. PyMC-based MMM modeling
3. Agent for forecasting, analysis, and optimization
4. Utilities for loading real MMM datasets
"""

from src.data_generation import ChannelConfig, DataGenerator, DataGeneratorConfig
from src.datasets import (
    load_he_mmm_dataset,
    get_media_channels,
    get_control_variables,
    prepare_mmm_data,
    summarize_dataset,
)
from src.model import ChannelSpec, MediaMixModel, MMModelConfig
from src.agent import MMAgent, ForecastRequest, HistoricalAnalysisResult
from src.optimizer import BudgetConstraints, BudgetOptimizer, OptimizationResult
from src.transforms import geometric_adstock, hill_saturation, logistic_saturation

__version__ = "0.1.0"

__all__ = [
    # Data generation
    "ChannelConfig",
    "DataGenerator",
    "DataGeneratorConfig",
    # Datasets
    "load_he_mmm_dataset",
    "get_media_channels",
    "get_control_variables",
    "prepare_mmm_data",
    "summarize_dataset",
    # Model
    "ChannelSpec",
    "MediaMixModel",
    "MMModelConfig",
    # Agent
    "MMAgent",
    "ForecastRequest",
    "HistoricalAnalysisResult",
    # Optimizer
    "BudgetConstraints",
    "BudgetOptimizer",
    "OptimizationResult",
    # Transforms
    "geometric_adstock",
    "hill_saturation",
    "logistic_saturation",
]
