# Databricks notebook source
"""
MMM Agent - Forecasting, Analysis, and Optimization

This notebook demonstrates using the MMM agent for:
1. Generating forecasts
2. Analyzing historical performance
3. Optimizing budget allocation
"""

# COMMAND ----------

# MAGIC %pip install -e .

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from src.model import MediaMixModel, MMModelConfig, ChannelSpec
from src.agent import MMAgent, ForecastRequest
from src.optimizer import BudgetConstraints
import pandas as pd
import arviz as az
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data and Model

# COMMAND ----------

# Load historical data
CATALOG = "main"
SCHEMA = "mmm"
TABLE = "synthetic_mmm_data"

spark = SparkSession.builder.getOrCreate()
df_spark = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE}")
df = df_spark.toPandas()
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

print(f"Loaded {len(df)} rows of historical data")

# COMMAND ----------

# Recreate model configuration (same as in 02_run_model.py)
channels = [
    ChannelSpec(
        name="tv",
        beta_prior_sigma=2.0,
        has_adstock=True,
        adstock_alpha_prior=5.0,
        adstock_beta_prior=2.0,
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

config = MMModelConfig(
    outcome_name="sales",
    intercept_mu=0.0,
    intercept_sigma=5.0,
    sigma_prior_beta=2.0,
    outcome_scale=100000,
    channels=channels
)

# COMMAND ----------

# Initialize model and load inference data
mmm = MediaMixModel(config)
mmm.idata = az.from_netcdf("/dbfs/mmm/models/latest_idata.nc")

# Scale data (needed for predictions)
mmm._scale_data(df)

print("✓ Model loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Agent

# COMMAND ----------

agent = MMAgent(model=mmm, data=df)
print("✓ Agent initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Historical Analysis

# COMMAND ----------

print("Analyzing historical performance...")
historical_results = agent.analyze_historical()

# Model fit metrics
print("\n=== Model Fit Metrics ===")
for metric, value in historical_results.model_fit_metrics.items():
    print(f"{metric}: {value:.2f}")

# COMMAND ----------

# Parameter summary (showing key parameters)
print("\n=== Key Parameter Estimates ===")
for param, stats in list(historical_results.summary_statistics.items())[:10]:
    print(f"\n{param}:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  SD: {stats['sd']:.3f}")
    print(f"  95% HDI: [{stats['hdi_3%']:.3f}, {stats['hdi_97%']:.3f}]")

# COMMAND ----------

# Channel insights
print("\n=== Channel Insights ===")
insights = agent.get_channel_insights()
for channel, insight in insights.items():
    print(f"\n{insight}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Forecasting

# COMMAND ----------

# Create future spend scenario
# Let's forecast for 30 days with different spend levels

import numpy as np

n_forecast = 30

future_spend = {
    "tv": [35000] * n_forecast,       # Increase TV spend
    "social": [20000] * n_forecast,   # Moderate social spend
    "search": [30000] * n_forecast,   # Moderate search spend
    "display": [15000] * n_forecast,  # Keep display moderate
}

forecast_request = ForecastRequest(future_spend=future_spend)

print("Generating forecasts...")
forecast_result = agent.forecast(forecast_request)

# COMMAND ----------

# Display forecast results
print("\n=== Forecast Results ===")
print(f"Number of forecast periods: {len(forecast_result.predictions)}")
print(f"\nPredicted sales:")
for i, (pred, lower, upper) in enumerate(zip(
    forecast_result.predictions[:5],
    forecast_result.lower_bound[:5],
    forecast_result.upper_bound[:5]
)):
    print(f"Period {i+1}: ${pred:,.0f} (95% CI: ${lower:,.0f} - ${upper:,.0f})")

# COMMAND ----------

# Visualize forecasts
import plotly.graph_objects as go

fig = go.Figure()

# Add forecast
periods = list(range(1, n_forecast + 1))
fig.add_trace(go.Scatter(
    x=periods,
    y=forecast_result.predictions,
    name="Forecast",
    line=dict(color="blue")
))

# Add confidence interval
fig.add_trace(go.Scatter(
    x=periods + periods[::-1],
    y=forecast_result.upper_bound + forecast_result.lower_bound[::-1],
    fill='toself',
    fillcolor='rgba(0,100,255,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='95% CI'
))

fig.update_layout(
    title="Sales Forecast with 95% Credible Interval",
    xaxis_title="Forecast Period",
    yaxis_title="Predicted Sales ($)",
    height=500
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Budget Optimization

# COMMAND ----------

# Define budget constraints
total_budget = 400000  # Total budget for optimization period

constraints = BudgetConstraints(
    total_budget=total_budget,
    min_spend_per_channel={
        "tv": 20000,
        "social": 10000,
        "search": 15000,
        "display": 5000
    },
    max_spend_per_channel={
        "tv": 150000,
        "social": 100000,
        "search": 120000,
        "display": 80000
    }
)

print(f"Optimizing budget allocation for ${total_budget:,}...")
optimization_result = agent.optimize_budget(constraints)

# COMMAND ----------

# Display optimization results
print("\n=== Optimal Budget Allocation ===")
for channel, allocation in optimization_result.optimal_allocation.items():
    pct = (allocation / total_budget) * 100
    print(f"{channel:12s}: ${allocation:>10,.0f} ({pct:>5.1f}%)")

print(f"\nExpected outcome: ${optimization_result.expected_outcome:,.0f}")

# COMMAND ----------

# Display channel ROAS
print("\n=== Channel ROAS ===")
for channel, roas in sorted(
    optimization_result.channel_roas.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{channel:12s}: {roas:.2f}x")

# COMMAND ----------

# Display recommendations
print("\n=== Recommendations ===")
for i, rec in enumerate(optimization_result.recommendations, 1):
    print(f"{i}. {rec}")

# COMMAND ----------

# Visualize optimal allocation
import plotly.express as px

allocation_df = pd.DataFrame([
    {"Channel": k, "Allocation": v, "ROAS": optimization_result.channel_roas[k]}
    for k, v in optimization_result.optimal_allocation.items()
])

fig = px.bar(
    allocation_df,
    x="Channel",
    y="Allocation",
    color="ROAS",
    title="Optimal Budget Allocation by Channel",
    labels={"Allocation": "Allocated Budget ($)", "ROAS": "ROAS"},
    color_continuous_scale="Viridis"
)
fig.update_layout(height=500)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario Comparison

# COMMAND ----------

# Compare different budget scenarios
scenarios = {
    "Current": 300000,
    "10% Increase": 330000,
    "20% Increase": 360000,
    "30% Increase": 390000,
}

results = []

for scenario_name, budget in scenarios.items():
    scenario_constraints = BudgetConstraints(
        total_budget=budget,
        min_spend_per_channel=constraints.min_spend_per_channel,
        max_spend_per_channel=constraints.max_spend_per_channel
    )
    
    result = agent.optimize_budget(scenario_constraints)
    results.append({
        "Scenario": scenario_name,
        "Budget": budget,
        "Expected Sales": result.expected_outcome,
        "ROAS": result.expected_outcome / budget
    })

scenario_df = pd.DataFrame(results)
print("\n=== Budget Scenario Comparison ===")
display(scenario_df)

# COMMAND ----------

# Visualize scenario comparison
fig = px.line(
    scenario_df,
    x="Budget",
    y="Expected Sales",
    markers=True,
    title="Expected Sales vs Budget",
    labels={"Budget": "Total Budget ($)", "Expected Sales": "Expected Sales ($)"}
)
fig.update_layout(height=500)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results to Delta

# COMMAND ----------

# Save optimization results
results_df = pd.DataFrame([
    {
        "timestamp": pd.Timestamp.now(),
        "total_budget": constraints.total_budget,
        "channel": channel,
        "allocation": allocation,
        "roas": optimization_result.channel_roas[channel]
    }
    for channel, allocation in optimization_result.optimal_allocation.items()
])

# Convert to Spark DataFrame and save
results_spark = spark.createDataFrame(results_df)
results_spark.write.format("delta").mode("append").saveAsTable(
    f"{CATALOG}.{SCHEMA}.optimization_results"
)

print(f"\n✓ Results saved to {CATALOG}.{SCHEMA}.optimization_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This agent provides three key capabilities:
# MAGIC 
# MAGIC 1. **Historical Analysis**: Understand past channel performance and model quality
# MAGIC 2. **Forecasting**: Predict future outcomes based on spend scenarios
# MAGIC 3. **Optimization**: Find optimal budget allocation to maximize outcomes
# MAGIC 
# MAGIC You can extend this agent with:
# MAGIC - LangChain/LangGraph integration for conversational interface
# MAGIC - Automated alerting for budget recommendations
# MAGIC - Integration with your marketing platforms for automated budget updates

