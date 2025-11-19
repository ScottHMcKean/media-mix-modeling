"""
Budget optimization module for MMM.

This module provides functionality to optimize marketing budget allocation
across channels based on fitted MMM models using response curves.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.optimize import minimize


class BudgetConstraints(BaseModel):
    """
    Budget constraints for optimization.

    Can be initialized with either:
    1. Separate min/max dicts: BudgetConstraints(total_budget=50000, min_spend_per_channel={...}, max_spend_per_channel={...})
    2. Nested channel dict: BudgetConstraints(total_budget=50000, channels={"facebook": {"min_spend": 1000, "max_spend": 20000}})
    """

    total_budget: float = Field(description="Total budget to allocate")
    min_spend_per_channel: Optional[Dict[str, float]] = Field(
        default=None, description="Minimum spend per channel"
    )
    max_spend_per_channel: Optional[Dict[str, float]] = Field(
        default=None, description="Maximum spend per channel"
    )
    channels: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Nested dict with min_spend and max_spend per channel: {'channel': {'min_spend': X, 'max_spend': Y}}",
    )

    def model_post_init(self, __context):
        """Convert nested channels dict to flat min/max dicts if provided."""
        if self.channels:
            # Extract from nested structure
            self.min_spend_per_channel = {
                ch: constraints.get("min_spend", 0) for ch, constraints in self.channels.items()
            }
            self.max_spend_per_channel = {
                ch: constraints.get("max_spend", self.total_budget)
                for ch, constraints in self.channels.items()
            }
        elif self.min_spend_per_channel is None or self.max_spend_per_channel is None:
            raise ValueError(
                "Must provide either 'channels' dict or both 'min_spend_per_channel' and 'max_spend_per_channel'"
            )


class OptimizationResult(BaseModel):
    """Results from budget optimization."""

    optimal_allocation: Dict[str, float] = Field(description="Optimal spend per channel")
    expected_outcome: float = Field(description="Expected outcome value (sales)")
    channel_contributions: Dict[str, float] = Field(description="Sales contribution per channel")
    channel_roas: Dict[str, float] = Field(description="ROAS per channel at optimal allocation")
    recommendations: List[str] = Field(description="Human-readable recommendations")


class BudgetOptimizer:
    """Optimize budget allocation based on fitted MMM response curves."""

    def __init__(self, model, data: pd.DataFrame):
        """
        Initialize optimizer with fitted model and historical data.

        Args:
            model: Fitted MediaMixModel instance
            data: Historical data (used for reference ranges)
        """
        self.model = model
        self.data = data
        self.channel_names = [c.name for c in model.config.channels]

        # Store historical spend ranges for each channel (for reference)
        self.spend_ranges = {}
        for channel in self.channel_names:
            self.spend_ranges[channel] = {
                "min": float(data[channel].min()),
                "max": float(data[channel].max()),
                "mean": float(data[channel].mean()),
            }

    def _calculate_contribution(
        self, spend: float, channel_name: str, posterior_params: Dict[str, float]
    ) -> float:
        """
        Calculate sales contribution for a given spend using response curves.

        This uses the model's internal scaling to properly evaluate the response curve.

        Args:
            spend: Raw spend value
            channel_name: Channel name
            posterior_params: Posterior mean parameters

        Returns:
            Sales contribution (scaled to actual sales units)
        """
        # Get channel spec
        channel_spec = next(c for c in self.model.config.channels if c.name == channel_name)

        # Use the model's scaler to normalize spend (0-1 range)
        # Model must be fitted before optimization
        spend_df = pd.DataFrame({channel_name: [spend]})
        spend_normalized = self.model.scalers[channel_name].transform(spend_df)[0, 0]
        spend_normalized = np.clip(spend_normalized, 0, 1)  # Ensure 0-1 range
        x = spend_normalized

        # Apply saturation curve (Hill transformation)
        if channel_spec.has_saturation:
            k = posterior_params[f"saturation_k_{channel_name}"]
            s = posterior_params[f"saturation_s_{channel_name}"]
            x = x**s / (k**s + x**s)

        # Apply beta (contribution coefficient)
        beta = posterior_params[f"beta_{channel_name}"]
        contribution = beta * x

        return contribution

    def _objective_function(
        self, allocation: np.ndarray, posterior_params: Dict[str, float]
    ) -> float:
        """
        Objective function to minimize (negative total sales).

        Args:
            allocation: Budget allocation array [spend_ch1, spend_ch2, ...]
            posterior_params: Posterior mean parameters

        Returns:
            Negative expected sales (for minimization)
        """
        # Start with base sales (intercept + trend at current time)
        base_sales = posterior_params["intercept"]

        # Add trend component if present (use end of observed period)
        if "beta_trend" in posterior_params:
            # Optimize for next period after observed data
            time_point = 1.0  # End of normalized time range
            base_sales += posterior_params["beta_trend"] * time_point

        total_sales = base_sales * self.model.config.outcome_scale

        # Add contribution from each channel
        for i, channel_name in enumerate(self.channel_names):
            spend = allocation[i]
            contribution = self._calculate_contribution(spend, channel_name, posterior_params)
            total_sales += contribution * self.model.config.outcome_scale

        # Return negative for minimization (we want to maximize sales)
        return -total_sales

    def optimize(self, constraints: BudgetConstraints, method: str = "SLSQP") -> OptimizationResult:
        """
        Optimize budget allocation to maximize expected sales.

        Args:
            constraints: Budget constraints (total, min, max per channel)
            method: Scipy optimization method (default: SLSQP for constrained optimization)

        Returns:
            Optimization results with allocation, ROAS, and recommendations
        """
        if self.model.idata is None:
            raise ValueError("Model must be fit before optimization")

        # Extract posterior means
        posterior_means = self.model.idata.posterior.mean(dim=["chain", "draw"])
        posterior_params = {"intercept": float(posterior_means["intercept"].values)}

        for channel_spec in self.model.config.channels:
            posterior_params[f"beta_{channel_spec.name}"] = float(
                posterior_means[f"beta_{channel_spec.name}"].values
            )

            if channel_spec.has_saturation:
                posterior_params[f"saturation_k_{channel_spec.name}"] = float(
                    posterior_means[f"saturation_k_{channel_spec.name}"].values
                )
                posterior_params[f"saturation_s_{channel_spec.name}"] = float(
                    posterior_means[f"saturation_s_{channel_spec.name}"].values
                )

        # Initial guess: equal allocation across channels
        x0 = np.array(
            [constraints.total_budget / len(self.channel_names) for _ in self.channel_names]
        )

        # Bounds: min/max spend per channel
        bounds = [
            (constraints.min_spend_per_channel[ch], constraints.max_spend_per_channel[ch])
            for ch in self.channel_names
        ]

        # Constraint: sum of allocations must equal total budget
        budget_constraint = {
            "type": "eq",
            "fun": lambda x: np.sum(x) - constraints.total_budget,
        }

        # Run optimization
        result = minimize(
            fun=lambda x: self._objective_function(x, posterior_params),
            x0=x0,
            method=method,
            bounds=bounds,
            constraints=[budget_constraint],
            options={"maxiter": 1000},
        )

        if not result.success:
            print(f"Warning: Optimization did not fully converge: {result.message}")

        # Extract results
        optimal_allocation = {ch: float(result.x[i]) for i, ch in enumerate(self.channel_names)}

        # Calculate contributions and ROAS at optimal allocation
        channel_contributions = {}
        channel_roas = {}

        for i, ch in enumerate(self.channel_names):
            spend = result.x[i]
            contribution_raw = self._calculate_contribution(spend, ch, posterior_params)
            contribution = contribution_raw * self.model.config.outcome_scale

            channel_contributions[ch] = contribution
            channel_roas[ch] = contribution / spend if spend > 0 else 0.0

        # Total expected sales (negative of objective)
        expected_outcome = -result.fun

        # Generate recommendations
        recommendations = self._generate_recommendations(
            optimal_allocation, channel_roas, channel_contributions, constraints
        )

        return OptimizationResult(
            optimal_allocation=optimal_allocation,
            expected_outcome=expected_outcome,
            channel_contributions=channel_contributions,
            channel_roas=channel_roas,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        allocation: Dict[str, float],
        roas: Dict[str, float],
        contributions: Dict[str, float],
        constraints: BudgetConstraints,
    ) -> List[str]:
        """
        Generate human-readable recommendations based on optimization.

        Args:
            allocation: Optimal spend allocation
            roas: ROAS per channel
            contributions: Sales contribution per channel
            constraints: Budget constraints

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        # Sort channels by ROAS
        sorted_by_roas = sorted(roas.items(), key=lambda x: x[1], reverse=True)
        top_channel, top_roas = sorted_by_roas[0]

        recommendations.append(
            f"Top performing channel: {top_channel.upper()} with ROAS of ${top_roas:.2f}"
        )

        # Check for channels at budget limits
        for ch, spend in allocation.items():
            max_spend = constraints.max_spend_per_channel[ch]
            if spend >= max_spend * 0.98:  # Within 2% of max
                recommendations.append(
                    f"{ch.upper()} is at maximum budget (${spend:,.0f}). "
                    f"Consider increasing limit to capture more potential."
                )

        # Check for underperforming channels
        mean_roas = np.mean(list(roas.values()))
        for ch, r in roas.items():
            if r < 1.0:  # ROAS < 1 means losing money
                recommendations.append(
                    f"WARNING: {ch.upper()} has ROAS < 1.0 (${r:.2f}). "
                    f"Consider reducing spend or improving efficiency."
                )
            elif r < mean_roas * 0.7:
                recommendations.append(
                    f"{ch.upper()} is underperforming (ROAS: ${r:.2f} vs avg ${mean_roas:.2f})"
                )

        # Budget efficiency
        total_contribution = sum(contributions.values())
        total_spend = sum(allocation.values())
        overall_roas = total_contribution / total_spend if total_spend > 0 else 0

        recommendations.append(
            f"Overall portfolio ROAS: ${overall_roas:.2f} "
            f"(${total_contribution:,.0f} incremental sales from ${total_spend:,.0f} spend)"
        )

        return recommendations
