# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Media Mix Modeling
# MAGIC ## 04: Real Data Example
# MAGIC
# MAGIC This notebook demonstrates the full workflow using real Home & Entertainment industry data,
# MAGIC showing compatibility with the original databricks-industry-solutions/media-mix-modeling approach.
# MAGIC
# MAGIC <div style="background-color: #d9f0ff; border-radius: 10px; padding: 15px; margin: 10px 0; font-family: Arial, sans-serif;">
# MAGIC   <strong>Note:</strong> This notebook has been tested on non-GPU accelerated serverless v4. <br/>
# MAGIC </div>

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from src.datasets import (
    load_he_mmm_dataset,
    get_media_channels,
    prepare_mmm_data,
    summarize_dataset,
)
from src.model import MediaMixModel, MMModelConfig, ChannelSpec
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Real Dataset
# MAGIC
# MAGIC This is a real-world Home & Entertainment MMM dataset with:
# MAGIC - Multiple media channels (TV, digital, social, etc.)
# MAGIC - Sales data
# MAGIC - Control variables (holidays, seasonality, economic indicators)

# COMMAND ----------

# Load the dataset
df_raw = load_he_mmm_dataset()

print(f"Loaded {len(df_raw)} weeks of data")
print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")

# Display summary
summary = summarize_dataset(df_raw)
print(f"\nDataset Summary:")
print(f"  Rows: {summary['n_rows']}")
print(f"  Columns: {summary['n_columns']}")
print(f"  Media channels (impressions): {summary['media_channels']['impressions']}")
print(f"  Media channels (spend): {summary['media_channels']['spend']}")
print(f"  Control variables:")
for k, v in summary["control_variables"].items():
    print(f"    - {k}: {v}")

# COMMAND ----------

# Display available media channels
spend_channels = get_media_channels(df_raw, prefix="mdsp")
print(f"\nAvailable spend channels ({len(spend_channels)}):")
for ch in spend_channels[:10]:  # Show first 10
    print(f"  - {ch}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data for Modeling
# MAGIC
# MAGIC Select media spend channels and sales outcome.

# COMMAND ----------

# Prepare data with spend channels
df_model = prepare_mmm_data(
    df_raw,
    outcome_col="sales",
    media_prefix="mdsp",  # Use media spend
    include_controls=[],  # No controls for this simple example
)

print(f"\nModel data shape: {df_model.shape}")
print(f"Columns: {list(df_model.columns)}")

display(df_model.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Model for Real Data
# MAGIC
# MAGIC We'll model a subset of channels for demonstration.

# COMMAND ----------

# Select top channels by spend variability
channel_cols = [col for col in df_model.columns if col != "sales"]
channel_variance = df_model[channel_cols].var().sort_values(ascending=False)

# Select top 5 channels
top_channels = channel_variance.head(5).index.tolist()
print(f"Modeling top 5 channels by spend variance:")
for ch in top_channels:
    print(f"  - {ch}: ${df_model[ch].mean():,.0f} avg/week")

# Create subset
df_subset = df_model[top_channels + ["sales"]].copy()

# COMMAND ----------

# Configure model for these channels
# Using generic priors suitable for normalized data
channels = []
for ch in top_channels:
    channels.append(
        ChannelSpec(
            name=ch,
            beta_prior_sigma=2.0,
            has_adstock=True,
            adstock_alpha_prior=4.0,
            adstock_beta_prior=3.0,
            has_saturation=True,
            saturation_k_prior_mean=0.5,
            saturation_s_prior_alpha=3.0,
            saturation_s_prior_beta=3.0,
        )
    )

# Estimate scale from data
outcome_scale = df_subset["sales"].std() * 10

config = MMModelConfig(
    outcome_name="sales",
    intercept_mu=0.0,
    intercept_sigma=5.0,
    sigma_prior_beta=2.0,
    outcome_scale=outcome_scale,
    channels=channels,
)

print(f"\n✓ Model configured with {len(channels)} channels")
print(f"  Outcome scale: {outcome_scale:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit Model
# MAGIC
# MAGIC Fit the Bayesian MMM using PyMC.
# MAGIC
# MAGIC **Note**: This uses production-quality sampling. For faster testing, reduce draws/tune/chains.

# COMMAND ----------

# Initialize and fit model
mmm = MediaMixModel(config)

print("Starting MCMC sampling...")
print("This may take several minutes...")

idata = mmm.fit(
    df=df_subset,
    draws=1000,  # Increase for production
    tune=500,  # Increase for production
    chains=2,  # Use 4 for production
    target_accept=0.95,
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

# Check R-hat
rhat_max = summary["r_hat"].max()
print(f"\nMax R-hat: {rhat_max:.4f}")
if rhat_max < 1.01:
    print("✓ Convergence looks good!")
else:
    print("⚠️ Consider more sampling iterations")

# COMMAND ----------

# Posterior distributions
fig = az.plot_posterior(idata, figsize=(15, 10))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Channel Contributions

# COMMAND ----------

# Calculate contributions
contributions_df = mmm.get_channel_contributions(df_subset)

print("Average Contribution by Channel:")
avg_contrib = contributions_df.mean().sort_values(ascending=False)
display(avg_contrib)

# COMMAND ----------

# Visualize contributions over time
import plotly.graph_objects as go

fig = go.Figure()
for channel in contributions_df.columns:
    fig.add_trace(
        go.Scatter(
            x=contributions_df.index,
            y=contributions_df[channel],
            name=channel.replace("mdsp_", ""),
            stackgroup="one",
        )
    )

fig.update_layout(
    title="Channel Contributions Over Time (Real Data)",
    xaxis_title="Date",
    yaxis_title="Contribution to Sales",
    height=600,
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results

# COMMAND ----------

# Save to MLflow
mmm.save_to_mlflow(experiment_name="/mmm/real_data_experiments", run_name="he_dataset_baseline")

print("\n✓ Results saved to MLflow")

# COMMAND ----------

# Save inference data
idata.to_netcdf("/dbfs/mmm/models/he_idata.nc")
print("✓ Inference data saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Insights

# COMMAND ----------

# Get channel insights from agent
from src.agent import MMAgent

agent = MMAgent(model=mmm, data=df_subset)
insights = agent.get_channel_insights()

print("=== Channel Insights ===\n")
for channel, insight in insights.items():
    print(insight)
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrates the complete workflow with **real data**:
# MAGIC
# MAGIC 1. ✓ Load real MMM dataset (Home & Entertainment example)
# MAGIC 2. ✓ Prepare and explore the data
# MAGIC 3. ✓ Configure model with appropriate priors
# MAGIC 4. ✓ Fit Bayesian MMM with PyMC
# MAGIC 5. ✓ Analyze results and channel contributions
# MAGIC 6. ✓ Save to MLflow for tracking
# MAGIC
# MAGIC This approach is fully compatible with the original
# MAGIC [databricks-industry-solutions/media-mix-modeling](https://github.com/databricks-industry-solutions/media-mix-modeling)
# MAGIC repository while providing a cleaner, more modular implementation.
