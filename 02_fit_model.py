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

import mlflow
from src.model import MediaMixModel

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

config.get('data')

# COMMAND ----------

# Load configuration using MLflow
config = mlflow.models.ModelConfig(
    development_config="example_config.yaml"
    )
ws_config = config.get("workspace")
data_config = config.get("data")
model_config = config.get("model")

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

df = spark.table(f"{ws_config['catalog']}.{ws_config['schema']}.{data_config['table']}").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Configure and Fit Model with MCMC
# MAGIC
# MAGIC Define the Bayesian model structure by specifying **priors** for each channel. Priors represent our beliefs before seeing the data:
# MAGIC
# MAGIC - **β (beta)**: Channel impact on sales - how much each $ of spend contributes
# MAGIC - **Adstock (α)**: Carryover decay rate - how long channel effects persist (e.g., TV ads have lasting impact)
# MAGIC - **Saturation (k, s)**: Diminishing returns parameters - efficiency drops at high spend levels
# MAGIC
# MAGIC The model will update these priors based on observed data to produce **posterior distributions** that quantify our updated beliefs with uncertainty bounds.
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

mmm = MediaMixModel.from_config(config.get("model"))
mmm.fit(
    df=df,
    draws=model_config["sampling"]["draws"],
    tune=model_config["sampling"]["tune"],
    chains=model_config["sampling"]["chains"],
    target_accept=model_config["sampling"]["target_accept"],
)

# COMMAND ----------

run_id = mmm.save_to_mlflow(
    experiment_name="/Workspace/Users/scott.mckean@databricks.com/experiments",
    register_model=True,
    model_name="mmm"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Diagnose Model Convergence & Performance
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
summary = az.summary(mmm.idata)
display(summary)

# COMMAND ----------

# Trace plots - should look like "fuzzy caterpillars"
print("Trace Plots (check for convergence):")
fig = az.plot_trace(mmm.idata, compact=True, figsize=(15, 10))
plt.tight_layout()
plt.show()

# COMMAND ----------

# Posterior distributions with uncertainty intervals
print("Posterior Distributions:")
fig = az.plot_posterior(mmm.idata, figsize=(15, 10))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Fit Quality

# COMMAND ----------

# WAIC and LOO
waic = az.waic(mmm.idata)
loo = az.loo(idata)

print(f"WAIC: {waic.elpd_waic:.2f} ± {waic.se:.2f}")
print(f"LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")

# R-hat (should be < 1.01)
rhat_summary = summary["r_hat"]
print(f"\nR-hat range: {rhat_summary.min():.4f} - {rhat_summary.max():.4f}")
if rhat_summary.max() > 1.01:
    print("⚠️ Warning: Some R-hat values > 1.01, consider more sampling")
else:
    print("✓ All R-hat values look good")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Analyze Channel Performance & Attribution
# MAGIC
# MAGIC Decompose sales into channel-specific contributions and calculate key performance metrics:
# MAGIC - **Base Sales**: Organic sales without any ad spend (intercept)
# MAGIC - **Channel Contributions**: Sales attributed to each marketing channel
# MAGIC - **ROAS**: Return on Ad Spend - revenue generated per $ spent
# MAGIC - **% of Total Sales**: Each channel's share of overall revenue
# MAGIC - **% of Incremental Sales**: Each channel's share of sales beyond baseline
# MAGIC
# MAGIC **Decision-Making Guide:**
# MAGIC - **High ROAS channels**: Efficient spend - consider increasing investment
# MAGIC - **Low ROAS + high saturation**: Consider reallocating to other channels
# MAGIC - **High adstock (α)**: Can maintain impact with lower sustained spend

# COMMAND ----------

# Calculate performance metrics
performance_summary = mmm.get_channel_performance_summary(df)
performance_summary = performance_summary.sort_values("total_contribution", ascending=False)

print("Channel Performance Summary:")
display(
    performance_summary.style.format(
        {
            "total_contribution": "${:,.0f}",
            "total_spend": "${:,.0f}",
            "roas": "{:.2f}",
            "pct_of_total_sales": "{:.1f}%",
            "pct_of_incremental_sales": "{:.1f}%",
        }
    )
)

# COMMAND ----------

# Calculate contributions over time
contributions_df = mmm.get_channel_contributions(df)

print("\nChannel Contributions Over Time (first 10 weeks):")
display(contributions_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualizations

# COMMAND ----------

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 1. Stacked area chart of contributions over time
fig1 = go.Figure()

# Define color scheme
colors = px.colors.qualitative.Set2

# Add base first (at the bottom)
fig1.add_trace(
    go.Scatter(
        x=contributions_df.index,
        y=contributions_df["base"],
        name="Base Sales",
        stackgroup="one",
        fillcolor=colors[0],
        line=dict(width=0.5, color=colors[0]),
    )
)

# Add each channel
for i, channel_spec in enumerate(mmm.config.channels):
    fig1.add_trace(
        go.Scatter(
            x=contributions_df.index,
            y=contributions_df[channel_spec.name],
            name=channel_spec.name.title(),
            stackgroup="one",
            fillcolor=colors[i + 1],
            line=dict(width=0.5, color=colors[i + 1]),
        )
    )

fig1.update_layout(
    title="Sales Attribution Over Time",
    xaxis_title="Week",
    yaxis_title="Sales ($)",
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig1.show()

# COMMAND ----------

# 2. ROAS Comparison Bar Chart
perf_channels = performance_summary[performance_summary["channel"] != "base"].copy()

fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=perf_channels["channel"],
        y=perf_channels["roas"],
        marker_color=colors[1 : len(perf_channels) + 1],
        text=perf_channels["roas"].round(2),
        textposition="outside",
    )
)

fig2.update_layout(
    title="Return on Ad Spend (ROAS) by Channel",
    xaxis_title="Channel",
    yaxis_title="ROAS ($)",
    height=400,
    showlegend=False,
)
fig2.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="red",
    annotation_text="Break-even (ROAS=1)",
    annotation_position="right",
)
fig2.show()

# COMMAND ----------

# 3. Sales Composition Pie Chart
fig3 = go.Figure()
fig3.add_trace(
    go.Pie(
        labels=performance_summary["channel"],
        values=performance_summary["total_contribution"],
        marker=dict(colors=colors[: len(performance_summary)]),
        textinfo="label+percent",
        textposition="outside",
        hole=0.3,
    )
)

fig3.update_layout(title="Sales Composition: Base vs Paid Channels", height=500, showlegend=True)
fig3.show()

# COMMAND ----------

# Print total sales breakdown
total_actual = df[mmm.config.outcome_name].sum()
total_attributed = contributions_df.sum().sum()
print(f"Total Actual Sales: ${total_actual:,.0f}")
print(f"Total Attributed Sales: ${total_attributed:,.0f}")
print(
    f"Base Sales: ${contributions_df['base'].sum():,.0f} ({contributions_df['base'].sum()/total_actual*100:.1f}%)"
)
print(
    f"Incremental Sales (from ads): ${(total_actual - contributions_df['base'].sum()):,.0f} ({(total_actual - contributions_df['base'].sum())/total_actual*100:.1f}%)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Temporal ROAS Analysis
# MAGIC
# MAGIC Compare model performance across time periods to understand ROAS evolution:
# MAGIC - **Recent Period**: Last 6 months before end date
# MAGIC - **Full Period**: Complete dataset
# MAGIC
# MAGIC This analysis reveals:
# MAGIC - Which channels are improving vs declining
# MAGIC - Saturation effects over time
# MAGIC - Optimal reallocation strategies

# COMMAND ----------

import pandas as pd

# Ensure index is datetime
if not pd.api.types.is_datetime64_any_dtype(df.index):
    df.index = pd.to_datetime(df.index)

end_date = df.index.max()
start_date = end_date - pd.DateOffset(months=12)
df_recent = df.loc[(df.index >= start_date) & (df.index <= end_date)]

# COMMAND ----------

# Initialize model for recent period
mmm_recent = MediaMixModel(mmm_config)

print("Fitting model on recent period...")
idata_recent = mmm_recent.fit(
    df=df_recent,
    draws=model_config["sampling"]["draws"],
    tune=sampling_config["tune"],
    chains=sampling_config["chains"],
    target_accept=sampling_config["target_accept"],
)

print("✓ Recent period model fitted!")

# COMMAND ----------

# Get performance for recent period
perf_recent = mmm_recent.get_channel_performance_summary(df_recent)
perf_full = mmm.get_channel_performance_summary(df)

print("Recent Period Performance:")
display(
    perf_recent.style.format(
        {
            "total_contribution": "${:,.0f}",
            "total_spend": "${:,.0f}",
            "roas": "{:.2f}",
            "pct_of_total_sales": "{:.1f}%",
            "pct_of_incremental_sales": "{:.1f}%",
        }
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROAS Evolution Analysis

# COMMAND ----------

# Compare ROAS between periods
roas_comparison = []

for channel_spec in mmm.config.channels:
    channel = channel_spec.name

    roas_recent = perf_recent[perf_recent["channel"] == channel]["roas"].values[0]
    roas_full = perf_full[perf_full["channel"] == channel]["roas"].values[0]
    pct_change = ((roas_full - roas_recent) / roas_recent) * 100

    roas_comparison.append(
        {
            "channel": channel,
            "roas_recent_period": roas_recent,
            "roas_full_period": roas_full,
            "change_pct": pct_change,
            "direction": "↑" if pct_change > 0 else "↓",
        }
    )

roas_comparison_df = pd.DataFrame(roas_comparison)

print("📊 ROAS EVOLUTION")
display(
    roas_comparison_df.style.format(
        {"roas_recent_period": "${:.2f}", "roas_full_period": "${:.2f}", "change_pct": "{:+.1f}%"}
    )
)

# Visualize ROAS comparison
fig = go.Figure()

channels = roas_comparison_df["channel"].values
recent_roas = roas_comparison_df["roas_recent_period"].values
full_roas = roas_comparison_df["roas_full_period"].values

fig.add_trace(
    go.Bar(
        name="Recent Period",
        x=channels,
        y=recent_roas,
        text=[f"${x:.2f}" for x in recent_roas],
        textposition="outside",
    )
)

fig.add_trace(
    go.Bar(
        name="Full Period",
        x=channels,
        y=full_roas,
        text=[f"${x:.2f}" for x in full_roas],
        textposition="outside",
    )
)

fig.update_layout(
    title="ROAS Comparison: Recent vs Full Period",
    xaxis_title="Channel",
    yaxis_title="ROAS ($)",
    barmode="group",
    height=400,
)
fig.show()

# COMMAND ----------

# Print insights
print("\n🔍 KEY INSIGHTS:")
for _, row in roas_comparison_df.iterrows():
    change_icon = "🚀" if row["change_pct"] > 5 else "📉" if row["change_pct"] < -5 else "📊"
    print(f"\n{change_icon} {row['channel'].upper()}")
    print(f"  Recent: ${row['roas_recent_period']:.2f}")
    print(f"  Full:   ${row['roas_full_period']:.2f}")
    print(f"  Change: {row['change_pct']:+.1f}% {row['direction']}")

    if row["change_pct"] > 10:
        print(f"  💡 Strong improvement - consider increasing investment")
    elif row["change_pct"] < -10:
        print(f"  ⚠️  Declining efficiency - review targeting/creative")
    else:
        print(f"  ✅ Stable performance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Forecast Channel Spend & ROAS
# MAGIC
# MAGIC Use ARIMA models to forecast channel spend 52 weeks ahead and project ROAS using fitted response curves.

# COMMAND ----------

from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forecast Spend with ARIMA

# COMMAND ----------


def forecast_spend_arima(df, channel, periods=52):
    """Forecast channel spend using ARIMA model."""
    # Prepare data
    ts = df[channel].values

    # Fit ARIMA model (1,1,1) as default for marketing spend data
    model = ARIMA(ts, order=(1, 1, 1))
    fitted = model.fit()

    # Forecast
    forecast = fitted.forecast(steps=periods)

    # Get confidence intervals
    forecast_obj = fitted.get_forecast(steps=periods)
    conf_int = forecast_obj.conf_int()

    return {
        "forecast": forecast,
        "lower": conf_int.iloc[:, 0].values,
        "upper": conf_int.iloc[:, 1].values,
        "model": fitted,
    }


# COMMAND ----------

# Forecast spend for each channel
print("Forecasting channel spend (52 weeks ahead)...")
forecast_results = {}

# Generate future dates
last_date = df.index.max()
future_dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=52, freq="W-MON")

for channel_spec in mmm.config.channels:
    channel = channel_spec.name
    print(f"  • {channel.capitalize()}...")
    forecast_results[channel] = forecast_spend_arima(df, channel, periods=52)

# Create forecast dataframe
forecast_df = pd.DataFrame(
    {
        "date": future_dates,
        **{
            channel_spec.name: forecast_results[channel_spec.name]["forecast"]
            for channel_spec in mmm.config.channels
        },
    }
)
forecast_df = forecast_df.set_index("date")

print(f"✓ Forecasted spend from {future_dates[0].date()} to {future_dates[-1].date()}")
display(forecast_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Project ROAS for Forecasted Spend

# COMMAND ----------

# Get response curve parameters from full model
posterior_means = idata.posterior.mean(dim=["chain", "draw"])

roas_forecast = []
for channel_spec in mmm.config.channels:
    channel = channel_spec.name

    # Get average forecasted spend
    avg_spend = forecast_df[channel].mean()

    # Calculate expected contribution using response curve
    # Normalize spend to 0-1 range (using max observed spend as reference)
    max_spend = df[channel].max()
    x_norm = avg_spend / max_spend

    # Apply saturation curve
    if f"saturation_k_{channel}" in posterior_means:
        k = float(posterior_means[f"saturation_k_{channel}"])
        s = float(posterior_means[f"saturation_s_{channel}"])
        beta = float(posterior_means[f"beta_{channel}"])

        # Hill saturation
        saturated = x_norm**s / (k**s + x_norm**s)

        # Expected contribution (scaled back)
        contribution_per_week = beta * saturated * model_config["outcome_scale"]

        # Projected ROAS
        projected_roas = contribution_per_week / avg_spend if avg_spend > 0 else 0
    else:
        projected_roas = 0

    # Get current ROAS for comparison
    current_roas = perf_full[perf_full["channel"] == channel]["roas"].values[0]

    roas_forecast.append(
        {
            "channel": channel,
            "avg_forecasted_spend": avg_spend,
            "current_roas": current_roas,
            "projected_roas": projected_roas,
            "expected_change_pct": ((projected_roas - current_roas) / current_roas) * 100,
        }
    )

    print(
        f"  • {channel.capitalize()}: Current ${current_roas:.2f} → Projected ${projected_roas:.2f}"
    )

roas_forecast_df = pd.DataFrame(roas_forecast)

print("\n📈 PROJECTED ROAS:")
display(
    roas_forecast_df.style.format(
        {
            "avg_forecasted_spend": "${:,.0f}",
            "current_roas": "${:.2f}",
            "projected_roas": "${:.2f}",
            "expected_change_pct": "{:+.1f}%",
        }
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Spend Forecast

# COMMAND ----------

# Visualize historical + forecasted spend
fig = make_subplots(
    rows=len(mmm.config.channels),
    cols=1,
    subplot_titles=[f"{c.name.title()} Spend" for c in mmm.config.channels],
    vertical_spacing=0.1,
)

for i, channel_spec in enumerate(mmm.config.channels, 1):
    channel = channel_spec.name

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[channel],
            mode="lines",
            name=f"{channel} (historical)",
            line=dict(color="blue"),
            showlegend=(i == 1),
        ),
        row=i,
        col=1,
    )

    # Forecasted data
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df[channel],
            mode="lines",
            name=f"{channel} (forecast)",
            line=dict(color="orange", dash="dash"),
            showlegend=(i == 1),
        ),
        row=i,
        col=1,
    )

    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_results[channel]["upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        ),
        row=i,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_results[channel]["lower"],
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(255, 165, 0, 0.2)",
            fill="tonexty",
            showlegend=(i == 1),
            name="95% CI" if i == 1 else None,
        ),
        row=i,
        col=1,
    )

    fig.update_yaxes(title_text="Spend ($)", row=i, col=1)

fig.update_xaxes(title_text="Date", row=len(mmm.config.channels), col=1)
fig.update_layout(height=300 * len(mmm.config.channels), title_text="Spend Forecast (52 Weeks)")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Forecast Results

# COMMAND ----------

# Save forecast results for the app
output_dir = "/dbfs/mmm/local_data/temporal_analysis"
import os

os.makedirs(output_dir, exist_ok=True)

# Save performance summaries (CSV for local development)
perf_recent.to_csv(f"{output_dir}/performance_recent.csv", index=False)
perf_full.to_csv(f"{output_dir}/performance_full.csv", index=False)

# Save ROAS comparison (CSV for local development)
roas_comparison_df.to_csv(f"{output_dir}/roas_comparison.csv", index=False)

# Save spend forecast (CSV for local development)
forecast_df.to_csv(f"{output_dir}/spend_forecast.csv")

# Save ROAS forecast (CSV for local development)
roas_forecast_df.to_csv(f"{output_dir}/roas_forecast.csv", index=False)

# Save inference data to volumes
model_artifacts_path = workspace.get("model_artifacts_volume", "/dbfs/mmm/models")
os.makedirs(model_artifacts_path, exist_ok=True)
idata_recent.to_netcdf(f"{model_artifacts_path}/inference_recent.nc")
idata.to_netcdf(f"{model_artifacts_path}/inference_full.nc")

print(f"✓ All results saved to local files")

# Save to Delta tables
catalog = workspace["catalog"]
schema = workspace["schema"]
tables = model_config.get("tables", {})

# Save ROAS comparison to Delta
roas_comparison_table = tables.get("roas_comparison", "roas_comparison")
roas_comparison_spark = spark.createDataFrame(roas_comparison_df)
roas_comparison_spark.write.mode("overwrite").saveAsTable(
    f"{catalog}.{schema}.{roas_comparison_table}"
)
print(f"✓ ROAS comparison saved to {catalog}.{schema}.{roas_comparison_table}")

# Save spend forecast to Delta
spend_forecast_table = tables.get("spend_forecast", "spend_forecast")
forecast_spark = spark.createDataFrame(forecast_df.reset_index())
forecast_spark.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{spend_forecast_table}")
print(f"✓ Spend forecast saved to {catalog}.{schema}.{spend_forecast_table}")

# Save ROAS forecast to Delta
roas_forecast_table = tables.get("roas_forecast", "roas_forecast")
roas_forecast_spark = spark.createDataFrame(roas_forecast_df)
roas_forecast_spark.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{roas_forecast_table}")
print(f"✓ ROAS forecast saved to {catalog}.{schema}.{roas_forecast_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecast Summary

# COMMAND ----------

print("=" * 80)
print("FORECAST SUMMARY")
print("=" * 80)

print("\n📈 SPEND FORECAST (Next 52 Weeks)")
print("-" * 80)
for channel_spec in mmm.config.channels:
    channel = channel_spec.name
    current_avg = df[channel].tail(26).mean()  # Last 6 months
    forecast_avg = forecast_df[channel].mean()
    change = ((forecast_avg - current_avg) / current_avg) * 100
    print(f"  {channel.upper()}")
    print(f"    Current avg (last 6mo): ${current_avg:,.0f}/week")
    print(f"    Forecast avg:           ${forecast_avg:,.0f}/week")
    print(f"    Change:                 {change:+.1f}%")

print("\n🎯 PROJECTED ROAS (Based on Forecasted Spend)")
print("-" * 80)
for _, row in roas_forecast_df.iterrows():
    print(f"  {row['channel'].upper()}")
    print(f"    Current ROAS:    ${row['current_roas']:.2f}")
    print(f"    Projected ROAS:  ${row['projected_roas']:.2f}")
    print(f"    Expected change: {row['expected_change_pct']:+.1f}%")

print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Save Model to MLflow
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
mmm.save_to_mlflow(experiment_name="/Workspace/Users/scott.mckean@databricks.com/experiments/mmm")

print("\n✓ Model saved to MLflow!")

# COMMAND ----------

# Also save inference data for use in agent
idata.to_netcdf("/dbfs/mmm/models/latest_idata.nc")
print("\n✓ Inference data saved to /dbfs/mmm/models/latest_idata.nc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC You now have a comprehensive Bayesian MMM with full posterior distributions, temporal ROAS analysis, and ARIMA-based forecasting! Use these insights to:
# MAGIC
# MAGIC 1. **Immediate Actions**:
# MAGIC    - Review ROAS evolution to identify declining channels
# MAGIC    - Compare recent vs full period performance
# MAGIC    - Analyze spend forecasts and projected ROAS
# MAGIC    - Validate results against business intuition
# MAGIC
# MAGIC 2. **Decision Making**:
# MAGIC    - Reallocate budget from declining to improving channels
# MAGIC    - Set spend caps on saturating channels
# MAGIC    - Increase investment in channels with improving ROAS
# MAGIC    - Adjust spend based on projected ROAS changes
# MAGIC
# MAGIC 3. **Advanced Analysis** (run `03_agent.py`):
# MAGIC    - Generate forecasts for different spending scenarios
# MAGIC    - Optimize budget allocation to maximize ROI
# MAGIC    - Explore "what-if" scenarios for strategic planning
# MAGIC
# MAGIC 4. **Interactive Analysis**:
# MAGIC    - Use the Streamlit app (`app.py`) to explore results interactively
# MAGIC    - Share visualizations and metrics with stakeholders
# MAGIC    - Monitor ROAS evolution over time
# MAGIC
# MAGIC 5. **Productionization**:
# MAGIC    - Schedule notebooks as jobs for regular model updates
# MAGIC    - Build dashboards from the output tables
# MAGIC    - Slice analysis by brand, region, or other dimensions using distributed compute
