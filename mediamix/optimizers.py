from scipy import optimize
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Literal, Callable, Any
from mediamix.transforms import geometric_adstock, hill_saturation


class Channel(BaseModel):
    """Single marketing channel configuration"""

    name: str
    type: Literal["digital", "traditional", "other"]
    adstock_params: Dict[str, float] = Field(
        default_factory=lambda: {"decay_rate": 0.5, "peak_delay": 0}
    )
    saturation_params: Dict[str, float] = Field(
        default_factory=lambda: {"half_max": 1.0, "slope": 1.0}
    )


class ChannelConstraints(BaseModel):
    """Constraints for a single marketing channel"""

    min_budget: float = Field(..., ge=0)
    max_budget: float = Field(...)
    current_budget: float = Field(..., ge=0)

    @validator("max_budget")
    def max_budget_must_exceed_min(cls, v, values):
        if "min_budget" in values and v < values["min_budget"]:
            raise ValueError("max_budget must be greater than min_budget")
        return v


class ChannelMetrics(BaseModel):
    """Performance metrics for a marketing channel"""

    roas: float
    cpa: Optional[float] = None
    saturation_point: Optional[float] = None
    efficiency_score: Optional[float] = None


class OptimizationConstraints(BaseModel):
    """Global optimization constraints"""

    total_budget: float = Field(..., ge=0)
    min_channel_allocation: Optional[float] = Field(default=0.05, ge=0, le=1)
    max_channel_allocation: Optional[float] = Field(default=0.5, ge=0, le=1)
    target_roas: Optional[float] = None


class OptimizationRequest(BaseModel):
    """Input for optimization queries"""

    objective: Literal["roas", "sales", "efficiency"]
    total_budget: float
    channel_constraints: Dict[str, ChannelConstraints]
    target_metrics: Optional[Dict[str, float]] = None


class OptimizationConfig(BaseModel):
    """Configuration for optimization runs"""

    objective: Literal["roas", "sales", "efficiency"]
    constraints: Dict[str, Dict[str, float]]
    seasonality: Optional[Dict[str, float]] = None
    target_metrics: Optional[Dict[str, float]] = None


class OptimizationResult(BaseModel):
    """Results from optimization run"""

    channel_allocations: Dict[str, float]
    predicted_metrics: Dict[str, ChannelMetrics]
    overall_metrics: ChannelMetrics
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None
    recommendations: List[str]


class MarketingOptimizer:
    """Core optimization engine"""

    def __init__(
        self,
        model,
        data,
        column_sets,
        model_params: Dict[str, Any],
        objective_fn: Callable,
    ):
        self.model_params = model_params
        self.model = model
        self.objective_fn = objective_fn
        self.column_sets = column_sets
        self.channel_constraints = {}

    def set_channel_constraints(self, constraints):
        """
        Set min/max budget constraints for channels

        Args:
            constraints: Dict of channel: {'min': float, 'max': float}
        """
        self.channel_constraints = constraints

    def optimize_budget(
        self,
        total_budget: float,
        constraints: Dict[str, Dict[str, float]],
        objective_fn: Callable,
    ) -> np.ndarray:
        """
        Optimize budget allocation using scipy
        """
        # Implementation using scipy.optimize
        pass
