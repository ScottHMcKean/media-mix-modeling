# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Media Mix Modeling
# MAGIC ## PyMC Modeling
# MAGIC
# MAGIC This notebook fits a Bayesian MMM using PyMC-Marketing and logs the model to MLflow.
# MAGIC
# MAGIC <div style="background-color: #d9f0ff; border-radius: 10px; padding: 15px; margin: 10px 0; font-family: Arial, sans-serif;">
# MAGIC   <strong>Note:</strong> This notebook has been tested on non-GPU accelerated serverless v4. <br/>
# MAGIC </div>

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install .
# MAGIC %restart_python

# COMMAND ----------

from src.model import MediaMixModel, MMModelConfig, ChannelSpec
import pandas as pd
import mlflow
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why Bayesian MMM?
# MAGIC
# MAGIC This accelerator uses a **Bayesian approach with PyMC** for several key advantages:
# MAGIC
# MAGIC - **Uncertainty quantification**: Get full posterior distributions, not just point estimates (e.g., β = 1.5 ± 0.1 vs 1.5 ± 1.0)
# MAGIC - **Interpretability**: Results are intuitive for decision-makers to understand and act upon
# MAGIC - **Flexible modeling**: Easily incorporate domain knowledge through priors and custom model structures
# MAGIC - **Pythonic & performant**: PyMC is easy to use, doesn't require a separate modeling language, and is built on fast vectorized libraries
# MAGIC
# MAGIC Alternative approaches include traditional ML models or MMM-specific libraries like [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing), [Robyn](https://facebookexperimental.github.io/Robyn/), or [Lightweight MMM](https://github.com/google/lightweight_mmm).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Configuration

# COMMAND ----------

# Load configuration using MLflow
CONFIG_PATH = "example_config.yaml"
config = mlflow.models.ModelConfig(development_config=CONFIG_PATH)
model_config = config.get("model")

print(f"Configuration loaded from {CONFIG_PATH}")
print(f"Random seed: {model_config['random_seed']}")
print(
    f"Source table: {model_config['catalog']}.{model_config['schema']}.{model_config['data_table']}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load Data from Gold Table
# MAGIC
# MAGIC Load aggregated marketing spend data from a [gold Delta table](https://www.databricks.com/glossary/medallion-architecture) in Unity Catalog. In production, this table would be the output of a data pipeline:
# MAGIC - **Bronze**: Raw data from marketing sources
# MAGIC - **Silver**: Cleansed and standardized data
# MAGIC - **Gold**: Aggregated daily spend by channel with sales outcomes
# MAGIC
# MAGIC Getting to a clean, well-structured dataset is critical for MMM success!

# COMMAND ----------

# Load data from Delta table using config values
table_path = f"{model_config['catalog']}.{model_config['schema']}.{model_config['data_table']}"
df_spark = spark.table(table_path)
df = df_spark.toPandas()

# Set date as index
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

print(f"Loaded {len(df)} rows from {table_path}")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Configure the Model
# MAGIC
# MAGIC Define the Bayesian model structure by specifying **priors** for each channel. Priors represent our beliefs before seeing the data:
# MAGIC
# MAGIC - **β (beta)**: Channel impact on sales - how much each $ of spend contributes
# MAGIC - **Adstock (α)**: Carryover decay rate - how long channel effects persist (e.g., TV ads have lasting impact)
# MAGIC - **Saturation (k, s)**: Diminishing returns parameters - efficiency drops at high spend levels
# MAGIC
# MAGIC The model will update these priors based on observed data to produce **posterior distributions** that quantify our updated beliefs with uncertainty bounds.

# COMMAND ----------

# Define channel specifications from config
channels = []
for channel_name, channel_config in model_config["channels"].items():
    channels.append(
        ChannelSpec(
            name=channel_name,
            beta_prior_sigma=channel_config["beta_prior_sigma"],
            has_adstock=channel_config["has_adstock"],
            adstock_alpha_prior=channel_config.get("adstock_alpha_prior"),
            adstock_beta_prior=channel_config.get("adstock_beta_prior"),
            has_saturation=channel_config["has_saturation"],
            saturation_k_prior_mean=channel_config["saturation_k_prior_mean"],
            saturation_s_prior_alpha=channel_config["saturation_s_prior_alpha"],
            saturation_s_prior_beta=channel_config["saturation_s_prior_beta"],
        )
    )

# Create model configuration
mmm_config = MMModelConfig(
    outcome_name=model_config["outcome_name"],
    intercept_mu=model_config["priors"]["intercept_mu"],
    intercept_sigma=model_config["priors"]["intercept_sigma"],
    sigma_prior_alpha=model_config["priors"]["sigma_alpha"],
    sigma_prior_beta=model_config["priors"]["sigma_beta"],
    outcome_scale=model_config["outcome_scale"],
    channels=channels,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Fit Model with MCMC
# MAGIC
# MAGIC Run **MCMC (Markov Chain Monte Carlo)** sampling to explore the posterior distribution of model parameters. This gives us not just point estimates, but full distributions with uncertainty:
# MAGIC
# MAGIC **Sampling Parameters:**
# MAGIC - `draws=1000`: Number of posterior samples per chain (more = better estimates, slower)
# MAGIC - `tune=500`: Warmup steps to calibrate the sampler
# MAGIC - `chains=2`: Independent MCMC chains (use 4+ for production to assess convergence)
# MAGIC - `target_accept=0.95`: Target acceptance rate (higher = more careful exploration)
# MAGIC
# MAGIC **Note**: For production, increase draws (2000+) and chains (4) for more reliable estimates.

# COMMAND ----------

# Initialize model
mmm = MediaMixModel(mmm_config)

# Fit model
sampling_config = model_config["sampling"]
print("Starting MCMC sampling...")
idata = mmm.fit(
    df=df,
    draws=sampling_config["draws"],
    tune=sampling_config["tune"],
    chains=sampling_config["chains"],
    target_accept=sampling_config["target_accept"],
)

print("\n✓ Model fitting complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Model Diagnostics & Convergence
# MAGIC
# MAGIC Check model quality and convergence before trusting results:
# MAGIC
# MAGIC **Key Diagnostics:**
# MAGIC - **R-hat** (< 1.01): Measures chain convergence - all chains should explore the same posterior
# MAGIC - **ESS (Effective Sample Size)**: Number of independent samples (aim for 400+)
# MAGIC - **Trace plots**: Should look like "fuzzy caterpillars" - stable, well-mixed chains
# MAGIC
# MAGIC **Parameter Interpretation:**
# MAGIC - **β (beta)**: Higher values = stronger sales impact per $ spent
# MAGIC - **Saturation**: Low saturation = room to grow; high saturation = consider reducing spend
# MAGIC - **Adstock (α)**: Higher α = longer decay/carryover effects

# COMMAND ----------

import arviz as az
import matplotlib.pyplot as plt

# Summary statistics
print("Parameter Summary:")
summary = az.summary(idata)
display(summary)

# COMMAND ----------

# Trace plots - should look like "fuzzy caterpillars"
print("Trace Plots (check for convergence):")
fig = az.plot_trace(idata, compact=True, figsize=(15, 10))
plt.tight_layout()
plt.show()

# COMMAND ----------

# Posterior distributions with uncertainty intervals
print("Posterior Distributions:")
fig = az.plot_posterior(idata, figsize=(15, 10))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Fit Quality

# COMMAND ----------

# WAIC and LOO
#waic = az.waic(idata)
#loo = az.loo(idata)

#print(f"WAIC: {waic.waic:.2f} ± {waic.se:.2f}")
#print(f"LOO: {loo.loo:.2f} ± {loo.se:.2f}")

# R-hat (should be < 1.01)
rhat_summary = summary["r_hat"]
print(f"\nR-hat range: {rhat_summary.min():.4f} - {rhat_summary.max():.4f}")
if rhat_summary.max() > 1.01:
    print("⚠️ Warning: Some R-hat values > 1.01, consider more sampling")
else:
    print("✓ All R-hat values look good")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Analyze Channel Contributions
# MAGIC
# MAGIC Decompose sales into channel-specific contributions to understand:
# MAGIC - Which channels drive the most impact
# MAGIC - How contributions vary over time
# MAGIC - Relative efficiency of your marketing spend
# MAGIC
# MAGIC **Decision-Making Guide:**
# MAGIC - **Low saturation + high β**: Room to invest more
# MAGIC - **High saturation + high β**: Maintain current levels
# MAGIC - **High saturation + low β**: Consider reallocating to other channels
# MAGIC - **High adstock (α)**: Can maintain impact with lower sustained spend

# COMMAND ----------

# Calculate contributions
contributions_df = mmm.get_channel_contributions(df)

print("Channel Contributions (first 10 rows):")
display(contributions_df.head(10))

# Plot contributions over time
import plotly.graph_objects as go

fig = go.Figure()
for channel in contributions_df.columns:
    fig.add_trace(
        go.Scatter(
            x=contributions_df.index, y=contributions_df[channel], name=channel, stackgroup="one"
        )
    )

fig.update_layout(
    title="Channel Contributions Over Time",
    xaxis_title="Date",
    yaxis_title="Contribution to Sales",
    height=500,
)
fig.show()

# COMMAND ----------

# Total contribution by channel
total_contributions = contributions_df.sum()
print("\nTotal Contribution by Channel:")
display(total_contributions.sort_values(ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Save Model to MLflow
# MAGIC
# MAGIC Log the model, parameters, metrics, and diagnostic plots to MLflow for:
# MAGIC - **Experiment tracking**: Compare different model configurations over time
# MAGIC - **Reproducibility**: Track all artifacts needed to reproduce results
# MAGIC - **Collaboration**: Share results and artifacts across your team
# MAGIC - **Productionization**: Easily promote models to production
# MAGIC
# MAGIC The inference data is saved separately for use in the agent (notebook 03).

# COMMAND ----------

# Set MLflow experiment
mlflow_config = model_config["mlflow"]
mlflow.set_experiment("/Workspace/Users/scott.mckean@databricks.com/experiments/mmm")

# Save model with all artifacts
mmm.save_to_mlflow(
    experiment_name="/Workspace/Users/scott.mckean@databricks.com/experiments/mmm"
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
# MAGIC You now have a fitted Bayesian MMM with full posterior distributions for all parameters! Use these insights to:
# MAGIC
# MAGIC 1. **Immediate Actions**:
# MAGIC    - Review model diagnostics (R-hat, ESS, trace plots)
# MAGIC    - If convergence issues exist, adjust priors or increase draws/chains
# MAGIC    - Validate results against business intuition
# MAGIC
# MAGIC 2. **Decision Making**:
# MAGIC    - Identify channels with low saturation and high β for increased investment
# MAGIC    - Consider reducing spend on highly saturated channels
# MAGIC    - Leverage adstock effects to optimize spending patterns over time
# MAGIC
# MAGIC 3. **Advanced Analysis** (run `03_agent.py`):
# MAGIC    - Generate forecasts for different spending scenarios
# MAGIC    - Optimize budget allocation to maximize ROI
# MAGIC    - Explore "what-if" scenarios for strategic planning
# MAGIC
# MAGIC 4. **Productionization**:
# MAGIC    - Schedule this notebook as a job for regular model updates
# MAGIC    - Build dashboards from the output tables for stakeholders
# MAGIC    - Slice analysis by brand, region, or other dimensions using distributed compute
