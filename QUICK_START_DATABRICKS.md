# Quick Start: Databricks Integration

## TL;DR

Your MMM code now works seamlessly in both local and Databricks environments with automatic detection. You can save to Delta tables and use MLflow Model Registry.

## What's New?

### ✅ Automatic Environment Detection
```python
from src.environment import has_spark

if has_spark():
    print("Running in Databricks - will use Delta tables")
else:
    print("Running locally - will use local files")
```

### ✅ Unified Data I/O
```python
from src.data_io import load_data, save_data

# Auto-detects: uses Delta in Databricks, local files otherwise
df = load_data(
    source="data.csv",
    catalog="main",
    schema="mmm",
    table="synthetic_data"
)

save_data(df, destination="output.csv", 
          catalog="main", schema="mmm", table="output")
```

### ✅ Enhanced MLflow Integration
```python
# Save with full artifacts and Model Registry
run_id = mmm.save_to_mlflow(
    experiment_name="/Users/name@company.com/mmm",
    register_model=True,
    model_name="mmm_production"
)

# Load anywhere
mmm = MediaMixModel.load_from_mlflow(run_id)
```

## Quick Examples

### Local Testing
```python
import mlflow
from src.data_generation import DataGenerator
from src.model import MediaMixModel

# Load config
config = mlflow.models.ModelConfig(development_config="example_config.yaml")

# Generate data (saves locally)
generator = DataGenerator.from_config(config)
df = generator.generate()
generator.save(df, file_path="local_data/data.csv", use_delta=False)

# Fit and save model
mmm = MediaMixModel(config)
mmm.fit(df)
run_id = mmm.save_to_mlflow(experiment_name="local_experiments")
```

### Databricks Production
```python
# Same code, automatically uses Delta!
import mlflow
from src.data_generation import DataGenerator
from src.model import MediaMixModel

config = mlflow.models.ModelConfig(development_config="example_config.yaml")
workspace = config.get("workspace")

# Generate data (automatically saves to Delta)
generator = DataGenerator.from_config(config)
df = generator.generate()
generator.save(df)  # Uses Delta automatically

# Fit and save to Model Registry
mmm = MediaMixModel.from_config(config.get("model"))
mmm.fit(df)
run_id = mmm.save_to_mlflow(
    experiment_name="/Production/mmm",
    register_model=True,
    model_name="mmm_prod"
)
```

## Installation

### Databricks Notebook
```python
%pip install uv
%sh uv pip install .
%restart_python
```

### Local Development
```bash
uv pip install -e ".[dev]"
```

## Test It

```bash
# Run all tests
uv run pytest tests/test_environment.py tests/test_data_io.py tests/test_mlflow_integration.py -v

# Run demo
uv run python demo_local_mlflow.py
```

## Configuration

Your existing `example_config.yaml` works in both environments:

```yaml
workspace:
  catalog: shm
  schema: mmm

data:
  table: synthetic_data
  # ...

model:
  data_table: synthetic_data
  # ...
```

## Key Variables

```python
from src.environment import IS_DATABRICKS, HAS_SPARK

# Module-level constants for quick checks
if IS_DATABRICKS:
    print("In Databricks")
    
if HAS_SPARK:
    print("Spark available")
```

## Best Practices

1. **Let it auto-detect**: Don't hardcode `use_delta=True/False`
   ```python
   # Good - portable
   df = load_data(source="data.csv", catalog="main", schema="mmm", table="data")
   
   # Less portable
   df = load_data("data.csv", use_delta=False)
   ```

2. **Always use MLflow**: Even locally for reproducibility
   ```python
   run_id = mmm.save_to_mlflow(experiment_name="experiments")
   ```

3. **Use Model Registry in production**:
   ```python
   mmm.save_to_mlflow(
       experiment_name="/Production/mmm",
       register_model=True,
       model_name="mmm_prod"
   )
   ```

## Common Patterns

### Pattern 1: Train Locally, Deploy to Databricks
```python
# Local: develop and test
df = load_data("local_data/data.csv", use_delta=False)
mmm.fit(df)
run_id = mmm.save_to_mlflow(experiment_name="experiments")

# Databricks: load and use
mmm = MediaMixModel.load_from_mlflow(run_id)
df = load_data(catalog="main", schema="mmm", table="production_data", use_delta=True)
predictions = mmm.predict(df)
```

### Pattern 2: Full Production Pipeline
```python
# In Databricks
from src.data_io import load_data, save_data
from src.model import MediaMixModel

# Load data from Delta
df = load_data(catalog="main", schema="mmm", table="marketing_data")

# Fit model
mmm = MediaMixModel(config)
mmm.fit(df)

# Save to MLflow with Model Registry
run_id = mmm.save_to_mlflow(
    experiment_name="/Production/mmm",
    register_model=True,
    model_name="mmm_latest"
)

# Calculate and save results to Delta
performance = mmm.get_channel_performance_summary(df)
save_data(performance, catalog="main", schema="mmm", table="performance")

contributions = mmm.get_channel_contributions(df)
save_data(contributions, catalog="main", schema="mmm", table="contributions")
```

## Documentation

- **Full Guide**: `DATABRICKS_INTEGRATION.md`
- **Changes**: `CHANGES.md`
- **Demo**: `demo_local_mlflow.py`

## Support

1. Check `DATABRICKS_INTEGRATION.md` for detailed usage
2. Run `demo_local_mlflow.py` to see it in action
3. Look at test files for more examples
4. Review module docstrings for API details

## What Didn't Change

✅ All existing code still works
✅ No breaking changes
✅ New features are opt-in
✅ Same configuration format
✅ Same model API

Your existing notebooks and scripts will continue to work without any modifications!

