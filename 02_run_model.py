# Databricks notebook source
"""
Fit PyMC MMM Model

This notebook fits a Bayesian MMM using PyMC and logs the model to MLflow.
"""

# COMMAND ----------

# MAGIC %pip install -e .

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from src.model import MediaMixModel, MMModelConfig, ChannelSpec
import pandas as pd
import mlflow
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Configure your catalog, schema, and table name
CATALOG = "main"
SCHEMA = "mmm"
TABLE = "synthetic_mmm_data"

# Load data from Delta table
spark = SparkSession.builder.getOrCreate()
df_spark = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE}")
df = df_spark.toPandas()

# Set date as index
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

print(f"Loaded {len(df)} rows")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Model

# COMMAND ----------

# Define channel specifications with priors
channels = [
    ChannelSpec(
        name="tv",
        beta_prior_sigma=2.0,
        has_adstock=True,
        adstock_alpha_prior=5.0,  # Beta distribution alpha
        adstock_beta_prior=2.0,   # Beta distribution beta
        has_saturation=True,
        saturation_k_prior_mean=0.5,
        saturation_s_prior_alpha=3.0,
        saturation_s_prior_beta=3.0
    ),
    ChannelSpec(
        name="social",
        beta_prior_sigma=2.0,
        has_adstock=True,
        adstock_alpha_prior=4.0,
        adstock_beta_prior=3.0,
        has_saturation=True,
        saturation_k_prior_mean=0.5,
        saturation_s_prior_alpha=3.0,
        saturation_s_prior_beta=3.0
    ),
    ChannelSpec(
        name="search",
        beta_prior_sigma=2.0,
        has_adstock=True,
        adstock_alpha_prior=4.5,
        adstock_beta_prior=2.5,
        has_saturation=True,
        saturation_k_prior_mean=0.5,
        saturation_s_prior_alpha=3.0,
        saturation_s_prior_beta=3.0
    ),
    ChannelSpec(
        name="display",
        beta_prior_sigma=2.0,
        has_adstock=True,
        adstock_alpha_prior=3.0,
        adstock_beta_prior=4.0,
        has_saturation=True,
        saturation_k_prior_mean=0.5,
        saturation_s_prior_alpha=3.0,
        saturation_s_prior_beta=3.0
    ),
]

# Create model configuration
config = MMModelConfig(
    outcome_name="sales",
    intercept_mu=0.0,
    intercept_sigma=5.0,
    sigma_prior_beta=2.0,
    outcome_scale=100000,  # Should match data generation scale
    channels=channels
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit Model
# MAGIC 
# MAGIC This will run MCMC sampling. Adjust sampling parameters based on your needs:
# MAGIC - `draws`: Number of posterior samples (more = better estimates, slower)
# MAGIC - `tune`: Number of tuning/warmup steps
# MAGIC - `chains`: Number of independent MCMC chains (4 is standard)

# COMMAND ----------

# Initialize model
mmm = MediaMixModel(config)

# Fit model
print("Starting MCMC sampling...")
idata = mmm.fit(
    df=df,
    draws=1000,    # Increase for production
    tune=500,      # Increase for production
    chains=2,      # Use 4 for production
    target_accept=0.95
)

print("\n✓ Model fitting complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Diagnostics

# COMMAND ----------

import arviz as az
import matplotlib.pyplot as plt

# Summary statistics
print("Parameter Summary:")
summary = az.summary(idata)
display(summary)

# COMMAND ----------

# Trace plots
fig = az.plot_trace(idata, compact=True, figsize=(15, 10))
plt.tight_layout()
plt.show()

# COMMAND ----------

# Posterior distributions
fig = az.plot_posterior(idata, figsize=(15, 10))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Fit Quality

# COMMAND ----------

# WAIC and LOO
waic = az.waic(idata)
loo = az.loo(idata)

print(f"WAIC: {waic.waic:.2f} ± {waic.se:.2f}")
print(f"LOO: {loo.loo:.2f} ± {loo.se:.2f}")

# R-hat (should be < 1.01)
rhat_summary = summary['r_hat']
print(f"\nR-hat range: {rhat_summary.min():.4f} - {rhat_summary.max():.4f}")
if rhat_summary.max() > 1.01:
    print("⚠️ Warning: Some R-hat values > 1.01, consider more sampling")
else:
    print("✓ All R-hat values look good")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Channel Contributions

# COMMAND ----------

# Calculate contributions
contributions_df = mmm.get_channel_contributions(df)

print("Channel Contributions (first 10 rows):")
display(contributions_df.head(10))

# Plot contributions over time
import plotly.graph_objects as go

fig = go.Figure()
for channel in contributions_df.columns:
    fig.add_trace(go.Scatter(
        x=contributions_df.index,
        y=contributions_df[channel],
        name=channel,
        stackgroup='one'
    ))

fig.update_layout(
    title="Channel Contributions Over Time",
    xaxis_title="Date",
    yaxis_title="Contribution to Sales",
    height=500
)
fig.show()

# COMMAND ----------

# Total contribution by channel
total_contributions = contributions_df.sum()
print("\nTotal Contribution by Channel:")
display(total_contributions.sort_values(ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Model to MLflow

# COMMAND ----------

# Set MLflow experiment
mlflow.set_experiment("/mmm/model_experiments")

# Save model
mmm.save_to_mlflow(
    experiment_name="/mmm/model_experiments",
    run_name="mmm_baseline_v1"
)

print("\n✓ Model saved to MLflow!")

# COMMAND ----------

# Also save inference data for use in agent
idata.to_netcdf("/dbfs/mmm/models/latest_idata.nc")
print("\n✓ Inference data saved to /dbfs/mmm/models/latest_idata.nc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review model diagnostics and fit quality
# MAGIC 2. If R-hat or other diagnostics are concerning, adjust priors or sampling parameters
# MAGIC 3. Run notebook `03_agent.py` to use the model for forecasting and optimization

