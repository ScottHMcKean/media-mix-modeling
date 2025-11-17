# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Media Mix Modeling
# MAGIC ## 03: Agent for Forecasting & Optimization
# MAGIC
# MAGIC This notebook demonstrates using the MMM agent for:
# MAGIC - Generating forecasts based on future spend scenarios
# MAGIC - Analyzing historical channel performance
# MAGIC - Optimizing budget allocation to maximize outcomes
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

from src.model import MediaMixModel, MMModelConfig, ChannelSpec
from src.agent import MMAgent, ForecastRequest
from src.optimizer import BudgetConstraints
import pandas as pd
import arviz as az
import mlflow
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Load configuration using MLflow
CONFIG_PATH = "example_config.yaml"
config = mlflow.models.ModelConfig(development_config=CONFIG_PATH)
agent_config = config.get("agent")
model_config = config.get("model")

print(f"Configuration loaded from {CONFIG_PATH}")
print(
    f"Historical data: {agent_config['catalog']}.{agent_config['schema']}.{agent_config['historical_data_table']}"
)
print(f"Model path: {agent_config['model_path']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data and Model

# COMMAND ----------

# Load historical data from config
spark = SparkSession.builder.getOrCreate()
table_path = (
    f"{agent_config['catalog']}.{agent_config['schema']}.{agent_config['historical_data_table']}"
)
df_spark = spark.table(table_path)
df = df_spark.toPandas()
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

print(f"Loaded {len(df)} rows of historical data from {table_path}")

# COMMAND ----------

# Recreate model configuration from config file
channels = []
for channel_name, channel_cfg in model_config["channels"].items():
    channels.append(
        ChannelSpec(
            name=channel_name,
            beta_prior_sigma=channel_cfg["beta_prior_sigma"],
            has_adstock=channel_cfg["has_adstock"],
            adstock_alpha_prior=channel_cfg.get("adstock_alpha_prior"),
            adstock_beta_prior=channel_cfg.get("adstock_beta_prior"),
            has_saturation=channel_cfg["has_saturation"],
            saturation_k_prior_mean=channel_cfg["saturation_k_prior_mean"],
            saturation_s_prior_alpha=channel_cfg["saturation_s_prior_alpha"],
            saturation_s_prior_beta=channel_cfg["saturation_s_prior_beta"],
        )
    )

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

# Initialize model and load inference data
mmm = MediaMixModel(mmm_config)
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

n_forecast = agent_config["forecasting"]["default_periods"]

# Define future spend scenarios from config
future_spend = {
    "adwords": [30000] * n_forecast,  # Increased AdWords spend
    "facebook": [20000] * n_forecast,  # Moderate Facebook spend
    "linkedin": [35000] * n_forecast,  # Increased LinkedIn spend
}

forecast_request = ForecastRequest(future_spend=future_spend)

print("Generating forecasts...")
forecast_result = agent.forecast(forecast_request)

# COMMAND ----------

# Display forecast results
print("\n=== Forecast Results ===")
print(f"Number of forecast periods: {len(forecast_result.predictions)}")
print(f"\nPredicted sales:")
for i, (pred, lower, upper) in enumerate(
    zip(
        forecast_result.predictions[:5],
        forecast_result.lower_bound[:5],
        forecast_result.upper_bound[:5],
    )
):
    print(f"Period {i+1}: ${pred:,.0f} (95% CI: ${lower:,.0f} - ${upper:,.0f})")

# COMMAND ----------

# Visualize forecasts
import plotly.graph_objects as go

fig = go.Figure()

# Add forecast
periods = list(range(1, n_forecast + 1))
fig.add_trace(
    go.Scatter(x=periods, y=forecast_result.predictions, name="Forecast", line=dict(color="blue"))
)

# Add confidence interval
fig.add_trace(
    go.Scatter(
        x=periods + periods[::-1],
        y=forecast_result.upper_bound + forecast_result.lower_bound[::-1],
        fill="toself",
        fillcolor="rgba(0,100,255,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
    )
)

fig.update_layout(
    title="Sales Forecast with 95% Credible Interval",
    xaxis_title="Forecast Period",
    yaxis_title="Predicted Sales ($)",
    height=500,
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Budget Optimization

# COMMAND ----------

# Define budget constraints from config
total_budget = agent_config["optimization"]["default_budget"]

# Build constraints from config
min_spend = {ch: cfg["min_spend"] for ch, cfg in agent_config["channels"].items()}
max_spend = {ch: cfg["max_spend"] for ch, cfg in agent_config["channels"].items()}

constraints = BudgetConstraints(
    total_budget=total_budget,
    min_spend_per_channel=min_spend,
    max_spend_per_channel=max_spend,
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
    optimization_result.channel_roas.items(), key=lambda x: x[1], reverse=True
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

allocation_df = pd.DataFrame(
    [
        {"Channel": k, "Allocation": v, "ROAS": optimization_result.channel_roas[k]}
        for k, v in optimization_result.optimal_allocation.items()
    ]
)

fig = px.bar(
    allocation_df,
    x="Channel",
    y="Allocation",
    color="ROAS",
    title="Optimal Budget Allocation by Channel",
    labels={"Allocation": "Allocated Budget ($)", "ROAS": "ROAS"},
    color_continuous_scale="Viridis",
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
        max_spend_per_channel=constraints.max_spend_per_channel,
    )

    result = agent.optimize_budget(scenario_constraints)
    results.append(
        {
            "Scenario": scenario_name,
            "Budget": budget,
            "Expected Sales": result.expected_outcome,
            "ROAS": result.expected_outcome / budget,
        }
    )

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
    labels={"Budget": "Total Budget ($)", "Expected Sales": "Expected Sales ($)"},
)
fig.update_layout(height=500)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results to Delta

# COMMAND ----------

# Save optimization results
results_df = pd.DataFrame(
    [
        {
            "timestamp": pd.Timestamp.now(),
            "total_budget": constraints.total_budget,
            "channel": channel,
            "allocation": allocation,
            "roas": optimization_result.channel_roas[channel],
        }
        for channel, allocation in optimization_result.optimal_allocation.items()
    ]
)

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
# MAGIC - DSPy-powered conversational interface (see `streamlit_app.py`)
# MAGIC - Automated alerting for budget recommendations
# MAGIC - Integration with your marketing platforms for automated budget updates
