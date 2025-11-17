"""
MMM Agent for forecasts, historical analysis, and optimization.

This module provides an agent interface powered by DSPy for interacting with
fitted MMM models to perform forecasting, analyze historical data, and optimize budgets.
"""

from typing import Dict, List, Optional

import arviz as az
import dspy
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.model import MediaMixModel
from src.optimizer import BudgetConstraints, BudgetOptimizer, OptimizationResult


# =============================================================================
# Pydantic Models for Agent Interface
# =============================================================================


class ForecastRequest(BaseModel):
    """Request for generating forecasts."""

    future_spend: Dict[str, List[float]] = Field(description="Future spend scenarios per channel")
    dates: Optional[List[str]] = Field(
        default=None, description="Optional dates for forecast periods"
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
    model_fit_metrics: Dict[str, float] = Field(description="Model fit statistics")
    summary_statistics: Dict[str, Dict[str, float]] = Field(
        description="Summary stats for parameters"
    )


# =============================================================================
# DSPy Signatures for Intelligent Analysis
# =============================================================================


class ChannelAnalysisSignature(dspy.Signature):
    """Analyze historical channel performance and provide insights."""

    channel_name: str = dspy.InputField(desc="Name of the marketing channel")
    historical_stats: str = dspy.InputField(desc="Historical statistics for the channel")
    model_parameters: str = dspy.InputField(
        desc="Fitted model parameters (beta, saturation, adstock)"
    )

    analysis: str = dspy.OutputField(desc="Detailed analysis of channel performance")
    insights: str = dspy.OutputField(desc="Key insights and recommendations")
    risk_factors: str = dspy.OutputField(desc="Potential risks or concerns")


class ForecastExplanationSignature(dspy.Signature):
    """Generate forecast explanation based on spend scenarios."""

    channels: str = dspy.InputField(desc="Channels and their future spend levels")
    predicted_outcome: str = dspy.InputField(desc="Predicted outcome with confidence intervals")
    historical_context: str = dspy.InputField(desc="Historical performance context")

    explanation: str = dspy.OutputField(desc="Clear explanation of the forecast")
    drivers: str = dspy.OutputField(desc="Key drivers of the predicted outcome")
    confidence_assessment: str = dspy.OutputField(desc="Assessment of forecast confidence")


class BudgetOptimizationSignature(dspy.Signature):
    """Provide optimization reasoning and recommendations."""

    total_budget: float = dspy.InputField(desc="Total available budget")
    current_allocation: str = dspy.InputField(desc="Current budget allocation by channel")
    optimal_allocation: str = dspy.InputField(desc="Optimized budget allocation by channel")
    channel_roas: str = dspy.InputField(desc="Return on ad spend by channel")

    reasoning: str = dspy.OutputField(desc="Reasoning behind the optimization")
    recommendations: str = dspy.OutputField(desc="Step-by-step recommendations")
    trade_offs: str = dspy.OutputField(desc="Trade-offs and considerations")


class GeneralMMMQuerySignature(dspy.Signature):
    """Answer general questions about MMM, marketing strategy, and data."""

    question: str = dspy.InputField(desc="User's question about MMM or marketing")
    context: str = dspy.InputField(desc="Relevant context from historical data and model")

    answer: str = dspy.OutputField(desc="Clear, concise answer to the question")
    supporting_evidence: str = dspy.OutputField(desc="Evidence or data supporting the answer")


# =============================================================================
# DSPy Modules
# =============================================================================


class MMMAnalyzer(dspy.Module):
    """DSPy module for analyzing channel performance."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(ChannelAnalysisSignature)

    def forward(self, channel_name: str, historical_stats: str, model_parameters: str):
        """Analyze a channel and return insights."""
        return self.analyze(
            channel_name=channel_name,
            historical_stats=historical_stats,
            model_parameters=model_parameters,
        )


class MMMForecaster(dspy.Module):
    """DSPy module for explaining forecasts."""

    def __init__(self):
        super().__init__()
        self.explain = dspy.ChainOfThought(ForecastExplanationSignature)

    def forward(self, channels: str, predicted_outcome: str, historical_context: str):
        """Generate forecast explanation."""
        return self.explain(
            channels=channels,
            predicted_outcome=predicted_outcome,
            historical_context=historical_context,
        )


class MMMOptimizer(dspy.Module):
    """DSPy module for budget optimization reasoning."""

    def __init__(self):
        super().__init__()
        self.optimize = dspy.ChainOfThought(BudgetOptimizationSignature)

    def forward(
        self,
        total_budget: float,
        current_allocation: str,
        optimal_allocation: str,
        channel_roas: str,
    ):
        """Generate optimization reasoning."""
        return self.optimize(
            total_budget=total_budget,
            current_allocation=current_allocation,
            optimal_allocation=optimal_allocation,
            channel_roas=channel_roas,
        )


class MMMAssistant(dspy.Module):
    """DSPy module for general MMM queries."""

    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought(GeneralMMMQuerySignature)

    def forward(self, question: str, context: str):
        """Answer general questions."""
        return self.answer(question=question, context=context)


# =============================================================================
# Main MMM Agent with DSPy Integration
# =============================================================================


class MMAgent:
    """
    Agent for interacting with MMM models, powered by DSPy for intelligent analysis.

    Provides both programmatic API and DSPy-powered natural language interface.
    """

    def __init__(
        self, model: MediaMixModel, data: pd.DataFrame, agent_config: Optional[dict] = None
    ):
        """
        Initialize agent.

        Args:
            model: Fitted MediaMixModel instance
            data: Historical data used for fitting
            agent_config: Optional agent configuration (for DSPy setup)
        """
        self.model = model
        self.data = data
        self.optimizer = BudgetOptimizer(
            model, channel_names=[c.name for c in model.config.channels]
        )

        # Initialize DSPy modules if config provided
        if agent_config:
            self._configure_dspy(agent_config)
            self.dspy_analyzer = MMMAnalyzer()
            self.dspy_forecaster = MMMForecaster()
            self.dspy_optimizer = MMMOptimizer()
            self.dspy_assistant = MMMAssistant()
        else:
            self.dspy_analyzer = None
            self.dspy_forecaster = None
            self.dspy_optimizer = None
            self.dspy_assistant = None

    def _configure_dspy(self, agent_config: dict):
        """
        Configure DSPy with Databricks Foundation Model API.

        Args:
            agent_config: Agent configuration dictionary with 'dspy' section
        """
        dspy_config = agent_config.get("dspy", {})

        # Configure DSPy with LM
        # For skeleton/demo, use OpenAI-compatible API
        # In production, configure with Databricks serving endpoints:
        # lm = dspy.LM(
        #     model='databricks/meta-llama-3-1-70b-instruct',
        #     api_base='https://<workspace>.databricks.com/serving-endpoints/...',
        #     api_key=dbutils.secrets.get(scope="...", key="...")
        # )
        
        model_name = dspy_config.get("llm_model", "openai/gpt-3.5-turbo")
        try:
            # Try to create LM with API key from environment
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "dummy-key-for-testing")
            lm = dspy.LM(
                model=model_name,
                api_key=api_key,
                max_tokens=dspy_config.get("max_tokens", 2048),
                temperature=dspy_config.get("temperature", 0.1),
            )
            dspy.configure(lm=lm)
        except Exception as e:
            # If LM configuration fails, continue without DSPy for now
            print(f"Warning: DSPy LM configuration failed: {e}")
            print("DSPy chat features will not be available.")

    # =========================================================================
    # Core Agent Methods (Programmatic API)
    # =========================================================================

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
            dates=request.dates,
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
            col: contributions_df[col].tolist() for col in contributions_df.columns
        }

        # Calculate model fit metrics
        waic = az.waic(self.model.idata)
        loo = az.loo(self.model.idata)

        model_fit_metrics = {
            "waic": float(waic.waic),
            "waic_se": float(waic.se),
            "loo": float(loo.loo),
            "loo_se": float(loo.se),
        }

        # Get parameter summary statistics
        summary = az.summary(self.model.idata)
        summary_statistics = {}

        for param in summary.index:
            summary_statistics[param] = {
                "mean": float(summary.loc[param, "mean"]),
                "sd": float(summary.loc[param, "sd"]),
                "hdi_3%": float(summary.loc[param, "hdi_3%"]),
                "hdi_97%": float(summary.loc[param, "hdi_97%"]),
            }

        return HistoricalAnalysisResult(
            channel_contributions=channel_contributions,
            model_fit_metrics=model_fit_metrics,
            summary_statistics=summary_statistics,
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
                insight_parts.append(
                    f"  - Carryover effect: {'Strong' if alpha > 0.7 else 'Moderate' if alpha > 0.4 else 'Weak'}"
                )

            if channel_spec.has_saturation:
                k = float(posterior_means[f"saturation_k_{channel_spec.name}"].values)
                s = float(posterior_means[f"saturation_s_{channel_spec.name}"].values)
                insight_parts.append(f"  - Half-saturation point: {k:.3f}")
                insight_parts.append(f"  - Saturation shape: {s:.3f}")
                insight_parts.append(
                    f"  - Diminishing returns: {'High' if s < 0.5 else 'Moderate' if s < 1.0 else 'Low'}"
                )

            insights[channel_spec.name] = "\n".join(insight_parts)

        return insights

    # =========================================================================
    # DSPy-Powered Natural Language Interface
    # =========================================================================

    def analyze_channel_with_dspy(self, channel_name: str, stats: dict, params: dict) -> dict:
        """
        Analyze a channel with DSPy intelligence.

        Args:
            channel_name: Name of the channel
            stats: Historical statistics
            params: Model parameters

        Returns:
            Dictionary with analysis, insights, and risk factors
        """
        if not self.dspy_analyzer:
            raise ValueError("DSPy not configured. Pass agent_config during initialization.")

        result = self.dspy_analyzer(
            channel_name=channel_name,
            historical_stats=str(stats),
            model_parameters=str(params),
        )

        return {
            "analysis": result.analysis,
            "insights": result.insights,
            "risk_factors": result.risk_factors,
        }

    def explain_forecast_with_dspy(self, channels: dict, prediction: dict, history: dict) -> dict:
        """
        Explain a forecast with DSPy intelligence.

        Args:
            channels: Future spend by channel
            prediction: Predicted outcome with confidence intervals
            history: Historical context

        Returns:
            Dictionary with explanation, drivers, and confidence assessment
        """
        if not self.dspy_forecaster:
            raise ValueError("DSPy not configured. Pass agent_config during initialization.")

        result = self.dspy_forecaster(
            channels=str(channels),
            predicted_outcome=str(prediction),
            historical_context=str(history),
        )

        return {
            "explanation": result.explanation,
            "drivers": result.drivers,
            "confidence": result.confidence_assessment,
        }

    def explain_optimization_with_dspy(
        self, budget: float, current: dict, optimal: dict, roas: dict
    ) -> dict:
        """
        Explain budget optimization with DSPy intelligence.

        Args:
            budget: Total budget
            current: Current allocation
            optimal: Optimal allocation
            roas: ROAS by channel

        Returns:
            Dictionary with reasoning, recommendations, and trade-offs
        """
        if not self.dspy_optimizer:
            raise ValueError("DSPy not configured. Pass agent_config during initialization.")

        result = self.dspy_optimizer(
            total_budget=budget,
            current_allocation=str(current),
            optimal_allocation=str(optimal),
            channel_roas=str(roas),
        )

        return {
            "reasoning": result.reasoning,
            "recommendations": result.recommendations,
            "trade_offs": result.trade_offs,
        }

    def chat(self, question: str, context: dict = None) -> dict:
        """
        Answer general questions with DSPy intelligence.

        Args:
            question: User's question
            context: Relevant context (optional)

        Returns:
            Dictionary with answer and supporting evidence
        """
        if not self.dspy_assistant:
            raise ValueError("DSPy not configured. Pass agent_config during initialization.")

        result = self.dspy_assistant(question=question, context=str(context or {}))

        return {"answer": result.answer, "evidence": result.supporting_evidence}
