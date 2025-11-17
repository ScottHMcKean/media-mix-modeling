"""
Budget optimization module for MMM.

This module provides functionality to optimize marketing budget allocation
across channels based on fitted MMM models.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.optimize import minimize


class BudgetConstraints(BaseModel):
    """Budget constraints for optimization."""

    total_budget: float = Field(description="Total budget to allocate")
    min_spend_per_channel: Dict[str, float] = Field(description="Minimum spend per channel")
    max_spend_per_channel: Dict[str, float] = Field(description="Maximum spend per channel")


class OptimizationResult(BaseModel):
    """Results from budget optimization."""

    optimal_allocation: Dict[str, float] = Field(description="Optimal spend per channel")
    expected_outcome: float = Field(description="Expected outcome value")
    channel_roas: Dict[str, float] = Field(description="ROAS per channel")
    recommendations: List[str] = Field(description="Human-readable recommendations")


class BudgetOptimizer:
    """Optimize budget allocation based on fitted MMM."""

    def __init__(self, model, channel_names: List[str]):
        """
        Initialize optimizer.

        Args:
            model: Fitted MediaMixModel instance
            channel_names: List of channel names
        """
        self.model = model
        self.channel_names = channel_names

    def _objective_function(
        self, allocation: np.ndarray, posterior_params: Dict[str, np.ndarray]
    ) -> float:
        """
        Objective function to maximize (negative for minimization).

        Args:
            allocation: Budget allocation array
            posterior_params: Posterior parameters from model

        Returns:
            Negative expected outcome
        """
        outcome = posterior_params["intercept"]

        for i, channel_name in enumerate(self.channel_names):
            spend = allocation[i]

            # Normalize spend
            spend_normalized = spend / 100000.0  # Simple normalization

            # Apply transformations based on channel config
            channel_spec = next(c for c in self.model.config.channels if c.name == channel_name)

            x = spend_normalized

            # Apply adstock
            if channel_spec.has_adstock:
                alpha = posterior_params.get(f"adstock_alpha_{channel_name}", 0.5)
                # Simplified adstock for optimization
                x = x * (1 / (1 - alpha))

            # Apply saturation
            if channel_spec.has_saturation:
                k = posterior_params.get(f"saturation_k_{channel_name}", 0.5)
                s = posterior_params.get(f"saturation_s_{channel_name}", 1.0)
                x = x**s / (k**s + x**s)

            # Add contribution
            beta = posterior_params[f"beta_{channel_name}"]
            outcome += beta * x

        # Return negative for minimization
        return -outcome * self.model.config.outcome_scale

    def optimize(self, constraints: BudgetConstraints, method: str = "SLSQP") -> OptimizationResult:
        """
        Optimize budget allocation.

        Args:
            constraints: Budget constraints
            method: Optimization method

        Returns:
            Optimization results
        """
        if self.model.idata is None:
            raise ValueError("Model must be fit before optimization")

        # Get posterior means
        posterior_means = self.model.idata.posterior.mean(dim=["chain", "draw"])
        posterior_params = {"intercept": float(posterior_means["intercept"].values)}

        for channel_spec in self.model.config.channels:
            posterior_params[f"beta_{channel_spec.name}"] = float(
                posterior_means[f"beta_{channel_spec.name}"].values
            )

            if channel_spec.has_adstock:
                posterior_params[f"adstock_alpha_{channel_spec.name}"] = float(
                    posterior_means[f"adstock_alpha_{channel_spec.name}"].values
                )

            if channel_spec.has_saturation:
                posterior_params[f"saturation_k_{channel_spec.name}"] = float(
                    posterior_means[f"saturation_k_{channel_spec.name}"].values
                )
                posterior_params[f"saturation_s_{channel_spec.name}"] = float(
                    posterior_means[f"saturation_s_{channel_spec.name}"].values
                )

        # Initial guess (equal allocation)
        x0 = np.array(
            [constraints.total_budget / len(self.channel_names) for _ in self.channel_names]
        )

        # Bounds for each channel
        bounds = [
            (constraints.min_spend_per_channel[ch], constraints.max_spend_per_channel[ch])
            for ch in self.channel_names
        ]

        # Constraint: sum of allocations = total budget
        constraint = {"type": "eq", "fun": lambda x: np.sum(x) - constraints.total_budget}

        # Optimize
        result = minimize(
            lambda x: self._objective_function(x, posterior_params),
            x0,
            method=method,
            bounds=bounds,
            constraints=[constraint],
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Create results
        optimal_allocation = {ch: float(result.x[i]) for i, ch in enumerate(self.channel_names)}

        expected_outcome = -result.fun

        # Calculate ROAS per channel
        channel_roas = {}
        for i, ch in enumerate(self.channel_names):
            spend = result.x[i]
            # Simplified ROAS calculation
            beta = posterior_params[f"beta_{ch}"]
            contribution = beta * (spend / 100000.0) * self.model.config.outcome_scale
            channel_roas[ch] = contribution / spend if spend > 0 else 0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            optimal_allocation, channel_roas, constraints
        )

        return OptimizationResult(
            optimal_allocation=optimal_allocation,
            expected_outcome=expected_outcome,
            channel_roas=channel_roas,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self, allocation: Dict[str, float], roas: Dict[str, float], constraints: BudgetConstraints
    ) -> List[str]:
        """
        Generate human-readable recommendations.

        Args:
            allocation: Optimal allocation
            roas: ROAS per channel
            constraints: Budget constraints

        Returns:
            List of recommendations
        """
        recommendations = []

        # Sort channels by ROAS
        sorted_channels = sorted(roas.items(), key=lambda x: x[1], reverse=True)

        recommendations.append(
            f"Top performing channel: {sorted_channels[0][0]} "
            f"(ROAS: {sorted_channels[0][1]:.2f})"
        )

        # Check if any channel is at max budget
        for ch, spend in allocation.items():
            if spend >= constraints.max_spend_per_channel[ch] * 0.95:
                recommendations.append(
                    f"Consider increasing max budget for {ch} - currently at limit"
                )

        # Check for underperforming channels
        mean_roas = np.mean(list(roas.values()))
        for ch, r in roas.items():
            if r < mean_roas * 0.5:
                recommendations.append(f"Channel {ch} is underperforming (ROAS: {r:.2f})")

        return recommendations
