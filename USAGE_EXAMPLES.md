# Usage Examples: Enhanced Features

This guide demonstrates the new features for MLflow configuration and performance analysis.

## Table of Contents
1. [MLflow Configuration from Config File](#mlflow-configuration-from-config-file)
2. [Saving Results to Delta Tables](#saving-results-to-delta-tables)
3. [Time-Based Performance Analysis](#time-based-performance-analysis)
4. [Complete Workflow Example](#complete-workflow-example)

---

## MLflow Configuration from Config File

### Configuration Setup

In your `example_config.yaml`:

```yaml
model:
  # ... other model settings ...
  
  # MLflow experiment tracking
  mlflow:
    experiment_name: /mmm/model_experiments
    run_name: mmm_baseline
    model_name: mmm_model
    register_model: true
  
  # Output tables (saved to Unity Catalog)
  tables:
    contributions: contributions
    performance_summary: performance_summary
    performance_analysis: performance_analysis
```

### Using MLflow Config

```python
import mlflow
from src.model import MediaMixModel
from src.data_io import load_data

# Load configuration
config = mlflow.models.ModelConfig(development_config="example_config.yaml")
model_config = config.get("model")
mlflow_config = model_config["mlflow"]

# Load data
df = load_data(
    source="synthetic_data.csv",
    catalog="main",
    schema="mmm",
    table="synthetic_data"
)

# Fit model
mmm = MediaMixModel(mmm_config)
mmm.fit(df, draws=2000, tune=1000, chains=4)

# Save to MLflow using config
run_id = mmm.save_to_mlflow(mlflow_config=mlflow_config)
# This automatically:
# - Uses experiment_name from config
# - Uses run_name from config
# - Registers model with model_name from config
# - Follows register_model setting from config

print(f"Model saved with run_id: {run_id}")
```

### Override Config Values

You can override config values with explicit parameters:

```python
# Override run_name but use other config values
run_id = mmm.save_to_mlflow(
    mlflow_config=mlflow_config,
    run_name="custom_run_name",  # Override
    # experiment_name, model_name, register_model come from config
)

# Or override everything
run_id = mmm.save_to_mlflow(
    experiment_name="/custom/experiment",
    run_name="custom_run",
    model_name="custom_model",
    register_model=True
)
```

---

## Saving Results to Delta Tables

### Save Contributions and Performance Summary

```python
from src.model import MediaMixModel
from src.data_io import load_data

# Load data
df = load_data(
    source="synthetic_data.csv",
    catalog="main",
    schema="mmm",
    table="synthetic_data"
)

# Fit model
mmm = MediaMixModel(mmm_config)
mmm.fit(df)

# Save contributions and performance summary to Delta
workspace = config.get("workspace")
tables = model_config.get("tables")

mmm.save_results_to_delta(
    df=df,
    catalog=workspace["catalog"],
    schema=workspace["schema"],
    contributions_table=tables["contributions"],
    performance_table=tables["performance_summary"],
    mode="overwrite"
)
```

This creates two Delta tables:
- `main.mmm.contributions` - Time series of channel contributions
- `main.mmm.performance_summary` - Overall performance metrics by channel

### Query Results from Delta

```python
# Query contributions
contributions_df = spark.table("main.mmm.contributions")
display(contributions_df)

# Query performance summary
performance_df = spark.table("main.mmm.performance_summary")
display(performance_df)
```

---

## Time-Based Performance Analysis

### Analyze Performance for Specific Time Periods

```python
# Get performance analysis for Q1 2023
q1_analysis = mmm.get_performance_analysis(
    df=df,
    start_date="2023-01-01",
    end_date="2023-03-31",
    analysis_name="q1_2023"
)

print(q1_analysis)
```

**Output Schema:**
```
| analysis_name | start_date | end_date   | channel  | total_contribution | total_spend | roas | pct_of_total_sales | pct_of_incremental_sales |
|---------------|------------|------------|----------|-------------------|-------------|------|--------------------|-----------------------|
| q1_2023       | 2023-01-01 | 2023-03-31 | adwords  | 125000            | 75000       | 1.67 | 15.2               | 22.5                      |
| q1_2023       | 2023-01-01 | 2023-03-31 | facebook | 95000             | 55000       | 1.73 | 11.5               | 17.1                      |
| ...           | ...        | ...        | ...      | ...               | ...         | ...  | ...                | ...                       |
```

### Save Performance Analysis to Delta

```python
# Analyze and save Q1 2023
mmm.save_performance_analysis_to_delta(
    df=df,
    catalog="main",
    schema="mmm",
    table="performance_analysis",
    start_date="2023-01-01",
    end_date="2023-03-31",
    analysis_name="q1_2023",
    mode="append"  # Append to existing table
)

# Analyze and save Q2 2023
mmm.save_performance_analysis_to_delta(
    df=df,
    catalog="main",
    schema="mmm",
    table="performance_analysis",
    start_date="2023-04-01",
    end_date="2023-06-30",
    analysis_name="q2_2023",
    mode="append"  # Append to same table
)

# Analyze and save Q3 2023
mmm.save_performance_analysis_to_delta(
    df=df,
    catalog="main",
    schema="mmm",
    table="performance_analysis",
    start_date="2023-07-01",
    end_date="2023-09-30",
    analysis_name="q3_2023",
    mode="append"
)

# Analyze and save Q4 2023
mmm.save_performance_analysis_to_delta(
    df=df,
    catalog="main",
    schema="mmm",
    table="performance_analysis",
    start_date="2023-10-01",
    end_date="2023-12-31",
    analysis_name="q4_2023",
    mode="append"
)
```

### Query Time-Based Analysis

```python
# Query all analyses
analysis_df = spark.table("main.mmm.performance_analysis")

# Compare ROAS across quarters for a specific channel
quarterly_roas = analysis_df.filter("channel = 'adwords'") \
    .select("analysis_name", "start_date", "end_date", "roas") \
    .orderBy("start_date")

display(quarterly_roas)
```

### Compare Performance Across Periods

```python
import pandas as pd
import plotly.express as px

# Load analysis from Delta
analysis_df = spark.table("main.mmm.performance_analysis").toPandas()

# Filter for paid channels (exclude base)
paid_channels = analysis_df[analysis_df["channel"] != "base"]

# Plot ROAS evolution
fig = px.bar(
    paid_channels,
    x="analysis_name",
    y="roas",
    color="channel",
    barmode="group",
    title="ROAS Evolution by Quarter",
    labels={"roas": "ROAS ($)", "analysis_name": "Quarter"}
)
fig.show()

# Plot spend evolution
fig2 = px.line(
    paid_channels,
    x="analysis_name",
    y="total_spend",
    color="channel",
    title="Spend Evolution by Quarter",
    labels={"total_spend": "Total Spend ($)", "analysis_name": "Quarter"}
)
fig2.show()
```

---

## Complete Workflow Example

### End-to-End Databricks Workflow

```python
# COMMAND ----------
# Install and setup
%pip install uv
%sh uv pip install .
%restart_python

# COMMAND ----------
# Imports
import mlflow
from src.data_generation import DataGenerator
from src.data_io import load_data, save_data
from src.model import MediaMixModel, MMMModelConfig, ChannelSpec

# COMMAND ----------
# Load configuration
CONFIG_PATH = "example_config.yaml"
config = mlflow.models.ModelConfig(development_config=CONFIG_PATH)

workspace = config.get("workspace")
model_config = config.get("model")
mlflow_config = model_config["mlflow"]
tables = model_config["tables"]

catalog = workspace["catalog"]
schema = workspace["schema"]

print(f"Workspace: {catalog}.{schema}")
print(f"Experiment: {mlflow_config['experiment_name']}")

# COMMAND ----------
# Generate synthetic data
generator = DataGenerator.from_config(config)
df = generator.generate()
generator.save(df, use_delta=True)  # Automatically saves to Delta
print(f"✓ Generated {len(df)} weeks of data")

# COMMAND ----------
# Load data from Delta
df = load_data(
    source="synthetic_data",
    catalog=catalog,
    schema=schema,
    table=model_config["data_table"],
    use_delta=True
)
print(f"✓ Loaded {len(df)} rows from Delta")

# COMMAND ----------
# Configure and fit model
channels = [
    ChannelSpec(
        name=ch_name,
        beta_prior_sigma=ch_config["beta_prior_sigma"],
        has_adstock=ch_config["has_adstock"],
        adstock_alpha_prior=ch_config.get("adstock_alpha_prior"),
        adstock_beta_prior=ch_config.get("adstock_beta_prior"),
        has_saturation=ch_config["has_saturation"],
        saturation_k_prior_mean=ch_config["saturation_k_prior_mean"],
        saturation_s_prior_alpha=ch_config["saturation_s_prior_alpha"],
        saturation_s_prior_beta=ch_config["saturation_s_prior_beta"],
    )
    for ch_name, ch_config in model_config["channels"].items()
]

mmm_config = MMMModelConfig(
    outcome_name=model_config["outcome_name"],
    outcome_scale=model_config["outcome_scale"],
    channels=channels,
    include_trend=model_config["include_trend"],
    trend_prior_sigma=model_config.get("trend_prior_sigma", 0.5),
)

mmm = MediaMixModel(mmm_config)

# Fit model
sampling = model_config["sampling"]
print("Fitting model...")
mmm.fit(
    df=df,
    draws=sampling["draws"],
    tune=sampling["tune"],
    chains=sampling["chains"],
    target_accept=sampling["target_accept"],
)
print("✓ Model fitted!")

# COMMAND ----------
# Save model to MLflow with Model Registry
run_id = mmm.save_to_mlflow(mlflow_config=mlflow_config)
print(f"✓ Model saved: {run_id}")

# COMMAND ----------
# Save contributions and performance to Delta
mmm.save_results_to_delta(
    df=df,
    catalog=catalog,
    schema=schema,
    contributions_table=tables["contributions"],
    performance_table=tables["performance_summary"],
    mode="overwrite"
)

# COMMAND ----------
# Perform time-based analyses
# Analyze each quarter separately
quarters = [
    ("q1_2023", "2023-01-01", "2023-03-31"),
    ("q2_2023", "2023-04-01", "2023-06-30"),
    ("q3_2023", "2023-07-01", "2023-09-30"),
    ("q4_2023", "2023-10-01", "2023-12-31"),
]

# Save first analysis (overwrite to create table)
mmm.save_performance_analysis_to_delta(
    df=df,
    catalog=catalog,
    schema=schema,
    table=tables["performance_analysis"],
    start_date=quarters[0][1],
    end_date=quarters[0][2],
    analysis_name=quarters[0][0],
    mode="overwrite"
)

# Append remaining analyses
for analysis_name, start_date, end_date in quarters[1:]:
    mmm.save_performance_analysis_to_delta(
        df=df,
        catalog=catalog,
        schema=schema,
        table=tables["performance_analysis"],
        start_date=start_date,
        end_date=end_date,
        analysis_name=analysis_name,
        mode="append"
    )

print("✓ Performance analyses saved")

# COMMAND ----------
# Query and visualize results
import plotly.express as px

# Load performance analysis
analysis_df = spark.table(f"{catalog}.{schema}.{tables['performance_analysis']}").toPandas()

# Filter for paid channels
paid_channels = analysis_df[analysis_df["channel"] != "base"]

# Plot ROAS by quarter
fig = px.bar(
    paid_channels,
    x="analysis_name",
    y="roas",
    color="channel",
    barmode="group",
    title="Quarterly ROAS Comparison",
    labels={"roas": "ROAS ($)", "analysis_name": "Quarter"}
)
fig.show()

# Display summary
print("\nQuarterly Performance Summary:")
display(paid_channels[["analysis_name", "channel", "roas", "total_spend", "total_contribution"]])

# COMMAND ----------
# Compare recent vs historical performance
# Analyze last 6 months
mmm.save_performance_analysis_to_delta(
    df=df,
    catalog=catalog,
    schema=schema,
    table=tables["performance_analysis"],
    start_date="2023-07-01",  # Last 6 months
    end_date="2023-12-31",
    analysis_name="recent_6mo",
    mode="append"
)

# Analyze full period
mmm.save_performance_analysis_to_delta(
    df=df,
    catalog=catalog,
    schema=schema,
    table=tables["performance_analysis"],
    analysis_name="full_period",
    mode="append"
)

print("✓ Comparison analyses saved")
```

---

## Local Testing

### Test Locally Before Deploying to Databricks

```python
import mlflow
from src.model import MediaMixModel
from src.data_io import load_data

# Load data locally
df = load_data("local_data/synthetic_data.csv", use_delta=False)

# Fit model
mmm = MediaMixModel(mmm_config)
mmm.fit(df, draws=500, tune=500, chains=2)

# Test MLflow config
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow_config = {
    "experiment_name": "local_testing",
    "run_name": "test_run",
    "model_name": "test_model",
    "register_model": False
}

run_id = mmm.save_to_mlflow(mlflow_config=mlflow_config)
print(f"Run ID: {run_id}")

# Test performance analysis
q1_analysis = mmm.get_performance_analysis(
    df=df,
    start_date="2020-01-01",
    end_date="2020-03-31",
    analysis_name="q1_2020"
)
print(q1_analysis)

# Save to local CSV for inspection
q1_analysis.to_csv("local_data/q1_analysis.csv", index=False)
```

---

## Best Practices

1. **Use Config Files**: Keep MLflow settings in config for consistency
   ```python
   run_id = mmm.save_to_mlflow(mlflow_config=mlflow_config)
   ```

2. **Append Mode for Time Series**: Use `mode="append"` for time-based analyses
   ```python
   mmm.save_performance_analysis_to_delta(..., mode="append")
   ```

3. **Descriptive Analysis Names**: Use clear names like `q1_2023` or `recent_6mo`
   ```python
   analysis_name="q1_2023"  # Good
   analysis_name="analysis1"  # Less clear
   ```

4. **Test Locally First**: Validate your workflow locally before deploying
   ```python
   # Local testing with small samples
   mmm.fit(df, draws=100, tune=100, chains=2)
   ```

5. **Query Delta Tables Efficiently**: Use Spark SQL for large datasets
   ```sql
   SELECT channel, analysis_name, roas, total_spend
   FROM main.mmm.performance_analysis
   WHERE analysis_name LIKE 'q%_2023'
   ORDER BY analysis_name, channel
   ```

---

## Troubleshooting

### Issue: "experiment_name is required"

**Solution**: Provide either mlflow_config or experiment_name
```python
# Option 1: Use config
run_id = mmm.save_to_mlflow(mlflow_config=mlflow_config)

# Option 2: Explicit parameter
run_id = mmm.save_to_mlflow(experiment_name="/my/experiment")
```

### Issue: "No data found between dates"

**Solution**: Check date format and range
```python
# Check data range
print(f"Data range: {df.index.min()} to {df.index.max()}")

# Ensure dates are within range
analysis = mmm.get_performance_analysis(
    df=df,
    start_date="2023-01-01",  # Must be >= df.index.min()
    end_date="2023-12-31"     # Must be <= df.index.max()
)
```

### Issue: Delta table schema conflicts

**Solution**: Use `mode="overwrite"` to recreate table
```python
mmm.save_performance_analysis_to_delta(
    ...,
    mode="overwrite"  # Recreates table with new schema
)
```

---

## Additional Resources

- **Main README**: Project overview and setup
- **QUICK_START_DATABRICKS.md**: Quick reference guide
- **Test Files**: See `tests/test_performance_analysis.py` for more examples
- **Config File**: See `example_config.yaml` for full configuration options

