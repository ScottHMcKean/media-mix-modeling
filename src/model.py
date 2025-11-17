"""
PyMC-based Media Mix Model.

This module provides functionality to build and fit MMM models using PyMC,
with MLflow integration for Databricks deployment.
"""

from typing import Dict, List, Optional, Tuple

import arviz as az
import mlflow
import numpy as np
import pandas as pd
import pymc as pm
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler

from src.transforms import geometric_adstock, hill_saturation


class ChannelSpec(BaseModel):
    """Specification for a media channel in the model."""

    name: str

    # Beta coefficient prior
    beta_prior_sigma: float = Field(default=1.0)

    # Adstock configuration
    has_adstock: bool = Field(default=False)
    adstock_alpha_prior: Optional[float] = Field(default=None)
    adstock_beta_prior: Optional[float] = Field(default=None)

    # Saturation configuration
    has_saturation: bool = Field(default=False)
    saturation_k_prior_mean: Optional[float] = Field(default=None)
    saturation_s_prior_alpha: Optional[float] = Field(default=None)
    saturation_s_prior_beta: Optional[float] = Field(default=None)


class MMModelConfig(BaseModel):
    """Configuration for MMM model."""

    outcome_name: str = "sales"

    # Intercept priors
    intercept_mu: float = 0.0
    intercept_sigma: float = 1.0

    # Noise prior
    sigma_prior_beta: float = 1.0

    # Outcome scaling
    outcome_scale: float = 1.0

    # Channels
    channels: List[ChannelSpec] = Field(default_factory=list)


class MediaMixModel:
    """Media Mix Model using PyMC."""

    def __init__(self, config: MMModelConfig):
        """
        Initialize MMM.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.idata = None
        self.scalers: Dict[str, MinMaxScaler] = {}

    def _scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale input data for modeling.

        Args:
            df: Input dataframe

        Returns:
            Scaled dataframe
        """
        df_scaled = df.copy()

        # Scale each channel to 0-1
        for channel in self.config.channels:
            self.scalers[channel.name] = MinMaxScaler()
            df_scaled[channel.name] = self.scalers[channel.name].fit_transform(df[[channel.name]])

        # Scale outcome
        df_scaled[self.config.outcome_name] = (
            df[self.config.outcome_name] / self.config.outcome_scale
        )

        return df_scaled

    def build_model(self, df: pd.DataFrame) -> pm.Model:
        """
        Build PyMC model.

        Args:
            df: Scaled input dataframe

        Returns:
            PyMC model instance
        """
        with pm.Model() as model:
            # Baseline intercept
            intercept = pm.Normal(
                "intercept", mu=self.config.intercept_mu, sigma=self.config.intercept_sigma
            )

            contributions = [intercept]

            # Add each channel
            for channel_spec in self.config.channels:
                x = df[channel_spec.name].values

                # Apply adstock if configured
                if channel_spec.has_adstock:
                    alpha = pm.Beta(
                        f"adstock_alpha_{channel_spec.name}",
                        alpha=channel_spec.adstock_alpha_prior,
                        beta=channel_spec.adstock_beta_prior,
                    )
                    x = geometric_adstock(x, alpha)

                # Apply saturation if configured
                if channel_spec.has_saturation:
                    k = pm.Gamma(
                        f"saturation_k_{channel_spec.name}",
                        mu=channel_spec.saturation_k_prior_mean,
                        sigma=0.5,
                    )
                    s = pm.Beta(
                        f"saturation_s_{channel_spec.name}",
                        alpha=channel_spec.saturation_s_prior_alpha,
                        beta=channel_spec.saturation_s_prior_beta,
                    )
                    x = hill_saturation(x, k, s)

                # Channel coefficient
                beta = pm.HalfNormal(
                    f"beta_{channel_spec.name}", sigma=channel_spec.beta_prior_sigma
                )

                contributions.append(beta * x)

            # Noise
            sigma = pm.HalfCauchy("sigma", beta=self.config.sigma_prior_beta)

            # Likelihood - sum all contributions
            # Start with intercept and add each channel contribution
            mu = intercept
            for i in range(1, len(contributions)):
                mu = mu + contributions[i]

            pm.Normal(
                self.config.outcome_name,
                mu=mu,
                sigma=sigma,
                observed=df[self.config.outcome_name].values,
            )

        self.model = model
        return model

    def fit(
        self,
        df: pd.DataFrame,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        return_inferencedata: bool = True,
    ) -> az.InferenceData:
        """
        Fit model using MCMC sampling.

        Args:
            df: Input dataframe
            draws: Number of samples to draw
            tune: Number of tuning steps
            chains: Number of MCMC chains
            target_accept: Target acceptance rate
            return_inferencedata: Whether to return InferenceData

        Returns:
            ArviZ InferenceData object
        """
        # Scale data
        df_scaled = self._scale_data(df)

        # Build model
        self.build_model(df_scaled)

        # Sample
        with self.model:
            self.idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=return_inferencedata,
            )

        return self.idata

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new data.

        Args:
            df: Input dataframe

        Returns:
            Array of predictions
        """
        if self.model is None or self.idata is None:
            raise ValueError("Model must be fit before making predictions")

        # Scale data using existing scalers
        df_scaled = df.copy()
        for channel in self.config.channels:
            df_scaled[channel.name] = self.scalers[channel.name].transform(df[[channel.name]])

        # Generate posterior predictive
        with self.model:
            pm.set_data(
                {channel.name: df_scaled[channel.name].values for channel in self.config.channels}
            )
            ppc = pm.sample_posterior_predictive(self.idata, predictions=True)

        # Rescale predictions
        predictions = (
            ppc.predictions[self.config.outcome_name].mean(axis=0) * self.config.outcome_scale
        )

        return predictions

    def get_channel_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate contribution of each channel to the outcome.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with channel contributions
        """
        if self.idata is None:
            raise ValueError("Model must be fit before calculating contributions")

        df_scaled = self._scale_data(df)
        contributions = {}

        # Get posterior means for each parameter
        posterior_means = self.idata.posterior.mean(dim=["chain", "draw"])

        for channel_spec in self.config.channels:
            x = df_scaled[channel_spec.name].values

            # Get beta coefficient
            beta_mean = posterior_means[f"beta_{channel_spec.name}"].values

            # Apply transformations (using mean parameters)
            if channel_spec.has_adstock:
                alpha_mean = posterior_means[f"adstock_alpha_{channel_spec.name}"].values
                weights = np.array([alpha_mean**i for i in range(12)])
                weights = weights / weights.sum()
                x = np.convolve(x, weights, mode="same")

            if channel_spec.has_saturation:
                k_mean = posterior_means[f"saturation_k_{channel_spec.name}"].values
                s_mean = posterior_means[f"saturation_s_{channel_spec.name}"].values
                x = x**s_mean / (k_mean**s_mean + x**s_mean)

            # Calculate contribution and rescale
            contributions[channel_spec.name] = beta_mean * x * self.config.outcome_scale

        return pd.DataFrame(contributions, index=df.index)

    def save_to_mlflow(self, experiment_name: str, run_name: Optional[str] = None):
        """
        Log model and results to MLflow.

        Args:
            experiment_name: MLflow experiment name
            run_name: Optional run name
        """
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            # Log configuration
            mlflow.log_dict(self.config.dict(), "config.json")

            # Log model metrics
            if self.idata is not None:
                summary = az.summary(self.idata)
                mlflow.log_dict(summary.to_dict(), "summary.json")

                # Log WAIC
                waic = az.waic(self.idata)
                mlflow.log_metrics({"waic": waic.waic, "waic_se": waic.se})

            # Save inference data
            if self.idata is not None:
                idata_path = "inference_data.nc"
                self.idata.to_netcdf(idata_path)
                mlflow.log_artifact(idata_path)

            print(f"Model logged to experiment: {experiment_name}")
