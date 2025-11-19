"""
PyMC-based Media Mix Model.

This module provides functionality to build and fit MMM models using PyMC,
with MLflow integration for Databricks deployment.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import arviz as az
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pymc as pm
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler

from src.environment import has_spark
from src.transforms import geometric_adstock, hill_saturation


class MMMPyFuncWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for MediaMixModel.

    This enables the model to be served and used for predictions through MLflow.
    """

    def __init__(self, model: "MediaMixModel"):
        """
        Initialize wrapper with fitted model.

        Args:
            model: Fitted MediaMixModel instance
        """
        self.config = model.config
        self.scalers = model.scalers

    def load_context(self, context):
        """
        Load model artifacts.

        Args:
            context: MLflow context with artifact paths
        """
        # Load inference data
        idata_path = context.artifacts["inference_data"]
        self.idata = az.from_netcdf(idata_path)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for input data.

        Args:
            context: MLflow context
            model_input: DataFrame with channel spend columns

        Returns:
            DataFrame with predictions
        """
        # This is a placeholder - full prediction logic would require
        # rebuilding the PyMC model which is complex in a serving context
        # For now, return expected sales based on posterior means
        raise NotImplementedError(
            "Prediction through PyFunc is not yet implemented. "
            "Use MediaMixModel.predict() directly or load with load_from_mlflow()."
        )


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


class MMMModelConfig(BaseModel):
    """Configuration for MMM model."""

    outcome_name: str = "sales"

    # Intercept priors
    intercept_mu: float = 0.0
    intercept_sigma: float = 1.0

    # Trend configuration
    include_trend: bool = Field(default=True, description="Include linear trend in base sales")
    trend_prior_sigma: float = Field(default=0.5, description="Prior std for trend coefficient")

    # Noise prior
    sigma_prior_beta: float = 1.0

    # Outcome scaling
    outcome_scale: float = 1.0

    # Channels
    channels: List[ChannelSpec] = Field(default_factory=list)


class MediaMixModel:
    """Media Mix Model using PyMC."""

    def __init__(self, config: MMMModelConfig):
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
        n_obs = len(df)

        with pm.Model() as model:
            # Baseline intercept
            intercept = pm.Normal(
                "intercept", mu=self.config.intercept_mu, sigma=self.config.intercept_sigma
            )

            # Base sales components
            base = intercept

            # Add trend if configured
            if self.config.include_trend:
                beta_trend = pm.Normal("beta_trend", mu=0, sigma=self.config.trend_prior_sigma)
                # Normalize time index to [0, 1] for stability
                time_index = np.arange(n_obs) / n_obs
                trend = beta_trend * time_index
                base = base + trend

            contributions = [base]

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
            # Start with base (intercept + trend) and add each channel contribution
            mu = base
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

            # Compute log likelihood for model comparison (WAIC, LOO)
            pm.compute_log_likelihood(self.idata)

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
        Calculate contribution of each channel to the outcome, including base sales.

        Args:
            df: Input dataframe with channel spend columns

        Returns:
            DataFrame with channel contributions including 'base' column
        """
        if self.idata is None:
            raise ValueError("Model must be fit before calculating contributions")

        df_scaled = self._scale_data(df)
        contributions = {}

        # Get posterior means for each parameter
        posterior_means = self.idata.posterior.mean(dim=["chain", "draw"])

        # Calculate base sales (intercept + trend)
        intercept_mean = posterior_means["intercept"].values
        base_contribution = np.full(len(df), intercept_mean)

        # Add trend if included in model
        if self.config.include_trend and "beta_trend" in posterior_means:
            beta_trend_mean = posterior_means["beta_trend"].values
            time_index = np.arange(len(df)) / len(df)
            base_contribution = base_contribution + beta_trend_mean * time_index

        contributions["base"] = base_contribution * self.config.outcome_scale

        # Calculate each channel's contribution
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

    def get_channel_performance_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive performance metrics for each channel.

        Args:
            df: Input dataframe with channel spend columns and outcome

        Returns:
            DataFrame with performance metrics: total_contribution, total_spend,
            roas, pct_of_total_sales, pct_of_incremental_sales
        """
        if self.idata is None:
            raise ValueError("Model must be fit before calculating performance")

        # Get contributions
        contributions_df = self.get_channel_contributions(df)

        # Get actual sales
        actual_sales = df[self.config.outcome_name].values
        total_sales = actual_sales.sum()

        # Base sales
        base_sales = contributions_df["base"].sum()
        incremental_sales = total_sales - base_sales

        # Calculate metrics for each channel
        metrics = []
        for channel_spec in self.config.channels:
            channel_name = channel_spec.name

            # Total contribution
            total_contribution = contributions_df[channel_name].sum()

            # Total spend
            total_spend = df[channel_name].sum()

            # ROAS (Return on Ad Spend)
            roas = total_contribution / total_spend if total_spend > 0 else 0

            # % of total sales
            pct_of_total = (total_contribution / total_sales * 100) if total_sales > 0 else 0

            # % of incremental sales (sales beyond base)
            pct_of_incremental = (
                (total_contribution / incremental_sales * 100) if incremental_sales > 0 else 0
            )

            metrics.append(
                {
                    "channel": channel_name,
                    "total_contribution": total_contribution,
                    "total_spend": total_spend,
                    "roas": roas,
                    "pct_of_total_sales": pct_of_total,
                    "pct_of_incremental_sales": pct_of_incremental,
                }
            )

        # Add base sales row
        metrics.append(
            {
                "channel": "base",
                "total_contribution": base_sales,
                "total_spend": 0,
                "roas": np.inf,  # No spend for base sales
                "pct_of_total_sales": (base_sales / total_sales * 100) if total_sales > 0 else 0,
                "pct_of_incremental_sales": 0,  # Base is not incremental
            }
        )

        return pd.DataFrame(metrics)

    def save_to_mlflow(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        register_model: bool = False,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Log model and results to MLflow with full artifact support.

        This saves:
        - Model configuration and parameters
        - Inference data (posterior samples) as NetCDF
        - Model diagnostics and metrics
        - PyFunc wrapper for predictions

        Args:
            experiment_name: MLflow experiment name
            run_name: Optional run name
            register_model: Whether to register model in Model Registry
            model_name: Model name for registration (required if register_model=True)

        Returns:
            MLflow run ID
        """
        if self.idata is None:
            raise ValueError("Model must be fit before saving to MLflow")

        if register_model and not model_name:
            raise ValueError("model_name is required when register_model=True")

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            # Log configuration
            mlflow.log_dict(self.config.model_dump(), "config.json")

            # Log sampling parameters
            mlflow.log_params(
                {
                    "n_channels": len(self.config.channels),
                    "outcome_name": self.config.outcome_name,
                    "outcome_scale": self.config.outcome_scale,
                    "include_trend": self.config.include_trend,
                }
            )

            # Log model diagnostics
            summary = az.summary(self.idata)

            # Log convergence metrics
            rhat_values = summary["r_hat"].values
            mlflow.log_metrics(
                {
                    "rhat_max": float(rhat_values.max()),
                    "rhat_mean": float(rhat_values.mean()),
                    "converged": float(rhat_values.max() < 1.01),
                }
            )

            # Log model comparison metrics
            try:
                waic = az.waic(self.idata)
                mlflow.log_metrics(
                    {
                        "elpd_waic": float(waic.elpd_waic),
                        "waic_se": float(waic.se),
                        "waic": float(waic.waic),
                    }
                )

                loo = az.loo(self.idata)
                mlflow.log_metrics(
                    {
                        "elpd_loo": float(loo.elpd_loo),
                        "loo_se": float(loo.se),
                        "loo": float(loo.loo),
                    }
                )
            except Exception as e:
                print(f"Warning: Could not compute WAIC/LOO: {e}")

            # Save artifacts in a temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save inference data (posterior samples)
                idata_path = Path(tmpdir) / "inference_data.nc"
                self.idata.to_netcdf(str(idata_path))
                mlflow.log_artifact(str(idata_path))

                # Save summary statistics
                summary_path = Path(tmpdir) / "summary.csv"
                summary.to_csv(summary_path)
                mlflow.log_artifact(str(summary_path))

                # Save model configuration
                config_path = Path(tmpdir) / "model_config.json"
                import json

                with open(config_path, "w") as f:
                    json.dump(self.config.model_dump(), f, indent=2)
                mlflow.log_artifact(str(config_path))

                # Log the model as a PyFunc
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=MMMPyFuncWrapper(self),
                    artifacts={"inference_data": str(idata_path)},
                    conda_env=self._get_conda_env(),
                )

            # Register model if requested
            if register_model:
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, model_name)
                print(f"✓ Model registered as: {model_name}")

            print(f"✓ Model logged to experiment: {experiment_name}")
            print(f"  Run ID: {run.info.run_id}")

            return run.info.run_id

    @staticmethod
    def _get_conda_env() -> dict:
        """Get conda environment for MLflow model."""
        return {
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.10",
                "pip",
                {
                    "pip": [
                        "pymc>=5.16.0",
                        "arviz>=0.20.0",
                        "pandas>=2.0.0",
                        "numpy>=1.24.0",
                        "scikit-learn>=1.6.1",
                        "pydantic==2.11.4",
                        "mlflow>=3.4.0",
                    ]
                },
            ],
        }

    @classmethod
    def load_from_mlflow(cls, run_id: str, artifact_path: str = "model") -> "MediaMixModel":
        """
        Load model from MLflow run.

        Args:
            run_id: MLflow run ID
            artifact_path: Path to model artifact (default: "model")

        Returns:
            MediaMixModel instance with loaded inference data
        """
        # Load artifacts
        client = mlflow.MlflowClient()
        artifact_uri = client.get_run(run_id).info.artifact_uri

        # Download inference data
        idata_uri = f"{artifact_uri}/inference_data.nc"
        local_path = mlflow.artifacts.download_artifacts(idata_uri)

        # Load config
        config_uri = f"{artifact_uri}/model_config.json"
        config_path = mlflow.artifacts.download_artifacts(config_uri)

        import json

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Reconstruct config
        channels = [ChannelSpec(**ch) for ch in config_dict["channels"]]
        config = MMMModelConfig(**{**config_dict, "channels": channels})

        # Create model instance
        model = cls(config)

        # Load inference data
        model.idata = az.from_netcdf(local_path)

        print(f"✓ Model loaded from run: {run_id}")

        return model
