"""
MMM Agent for forecasts, historical analysis, and optimization.

This module provides an agent interface for interacting with fitted MMM models
to perform forecasting, analyze historical data, and optimize budgets.
"""

from typing import Dict, List, Optional

import arviz as az
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.model import MediaMixModel
from src.optimizer import BudgetConstraints, BudgetOptimizer, OptimizationResult


class ForecastRequest(BaseModel):
    """Request for generating forecasts."""
    
    future_spend: Dict[str, List[float]] = Field(
        description="Future spend scenarios per channel"
    )
    dates: Optional[List[str]] = Field(
        default=None,
        description="Optional dates for forecast periods"
    )


class ForecastResult(BaseModel):
    """Results from forecast."""
    
    predictions: List[float] = Field(description="Predicted outcome values")
    lower_bound: List[float] = Field(description="Lower credible interval")
    upper_bound: List[float] = Field(description="Upper credible interval")
    dates: Optional[List[str]] = Field(default=None)


class HistoricalAnalysisResult(BaseModel):
    """Results from historical analysis."""
    
    channel_contributions: Dict[str, List[float]] = Field(
        description="Contribution of each channel over time"
    )
    model_fit_metrics: Dict[str, float] = Field(
        description="Model fit statistics"
    )
    summary_statistics: Dict[str, Dict[str, float]] = Field(
        description="Summary stats for parameters"
    )


class MMAgent:
    """Agent for interacting with MMM models."""
    
    def __init__(self, model: MediaMixModel, data: pd.DataFrame):
        """
        Initialize agent.
        
        Args:
            model: Fitted MediaMixModel instance
            data: Historical data used for fitting
        """
        self.model = model
        self.data = data
        self.optimizer = BudgetOptimizer(
            model,
            channel_names=[c.name for c in model.config.channels]
        )
    
    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """
        Generate forecasts for future spend scenarios.
        
        Args:
            request: Forecast request
            
        Returns:
            Forecast results with credible intervals
        """
        if self.model.idata is None:
            raise ValueError("Model must be fit before forecasting")
        
        # Create dataframe from future spend
        future_df = pd.DataFrame(request.future_spend)
        
        # Get predictions
        predictions = self.model.predict(future_df)
        
        # Calculate credible intervals using posterior samples
        # For simplicity, we'll use the posterior predictive distribution
        with self.model.model:
            # This is a simplified version - in practice you'd want full posterior predictive
            posterior_samples = self.model.idata.posterior
            
            # Get parameter samples
            intercept_samples = posterior_samples["intercept"].values.flatten()
            
            # Calculate outcome distribution
            outcomes = []
            for _ in range(100):  # Sample 100 predictions
                sample_idx = np.random.randint(len(intercept_samples))
                outcome = intercept_samples[sample_idx]
                
                for channel_spec in self.model.config.channels:
                    beta_samples = posterior_samples[f"beta_{channel_spec.name}"].values.flatten()
                    outcome += beta_samples[sample_idx] * future_df[channel_spec.name].mean()
                
                outcomes.append(outcome * self.model.config.outcome_scale)
            
            lower_bound = [np.percentile(outcomes, 2.5)] * len(predictions)
            upper_bound = [np.percentile(outcomes, 97.5)] * len(predictions)
        
        return ForecastResult(
            predictions=predictions.tolist(),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dates=request.dates
        )
    
    def analyze_historical(self) -> HistoricalAnalysisResult:
        """
        Analyze historical data and model fit.
        
        Returns:
            Historical analysis results
        """
        if self.model.idata is None:
            raise ValueError("Model must be fit before analysis")
        
        # Get channel contributions
        contributions_df = self.model.get_channel_contributions(self.data)
        channel_contributions = {
            col: contributions_df[col].tolist()
            for col in contributions_df.columns
        }
        
        # Calculate model fit metrics
        waic = az.waic(self.model.idata)
        loo = az.loo(self.model.idata)
        
        model_fit_metrics = {
            "waic": float(waic.waic),
            "waic_se": float(waic.se),
            "loo": float(loo.loo),
            "loo_se": float(loo.se)
        }
        
        # Get parameter summary statistics
        summary = az.summary(self.model.idata)
        summary_statistics = {}
        
        for param in summary.index:
            summary_statistics[param] = {
                "mean": float(summary.loc[param, "mean"]),
                "sd": float(summary.loc[param, "sd"]),
                "hdi_3%": float(summary.loc[param, "hdi_3%"]),
                "hdi_97%": float(summary.loc[param, "hdi_97%"])
            }
        
        return HistoricalAnalysisResult(
            channel_contributions=channel_contributions,
            model_fit_metrics=model_fit_metrics,
            summary_statistics=summary_statistics
        )
    
    def optimize_budget(self, constraints: BudgetConstraints) -> OptimizationResult:
        """
        Optimize budget allocation across channels.
        
        Args:
            constraints: Budget constraints
            
        Returns:
            Optimization results
        """
        return self.optimizer.optimize(constraints)
    
    def get_channel_insights(self) -> Dict[str, str]:
        """
        Get human-readable insights about channel performance.
        
        Returns:
            Dictionary of channel insights
        """
        if self.model.idata is None:
            raise ValueError("Model must be fit before getting insights")
        
        insights = {}
        posterior_means = self.model.idata.posterior.mean(dim=["chain", "draw"])
        
        for channel_spec in self.model.config.channels:
            beta = float(posterior_means[f"beta_{channel_spec.name}"].values)
            
            insight_parts = [f"Channel {channel_spec.name}:"]
            insight_parts.append(f"  - Base contribution coefficient: {beta:.3f}")
            
            if channel_spec.has_adstock:
                alpha = float(posterior_means[f"adstock_alpha_{channel_spec.name}"].values)
                insight_parts.append(f"  - Adstock decay rate: {alpha:.3f}")
                insight_parts.append(f"  - Carryover effect: {'Strong' if alpha > 0.7 else 'Moderate' if alpha > 0.4 else 'Weak'}")
            
            if channel_spec.has_saturation:
                k = float(posterior_means[f"saturation_k_{channel_spec.name}"].values)
                s = float(posterior_means[f"saturation_s_{channel_spec.name}"].values)
                insight_parts.append(f"  - Half-saturation point: {k:.3f}")
                insight_parts.append(f"  - Saturation shape: {s:.3f}")
                insight_parts.append(f"  - Diminishing returns: {'High' if s < 0.5 else 'Moderate' if s < 1.0 else 'Low'}")
            
            insights[channel_spec.name] = "\n".join(insight_parts)
        
        return insights


# Note: For LangGraph/LangChain agent integration, we can create tools from these methods
def create_langchain_tools(agent: MMAgent):
    """
    Create LangChain tools from agent methods.
    
    Args:
        agent: MMAgent instance
        
    Returns:
        List of LangChain tools
    """
    # This would integrate with LangChain/LangGraph for conversational agent
    # For now, just return the agent itself
    # In a full implementation, wrap each method as a BaseTool
    return agent

