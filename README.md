# Media Mix Modeling with PyMC and Databricks

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)
[![DBR](https://img.shields.io/badge/RUNTIME-ML-red?style=for-the-badge)](https://databricks.com/try-databricks)
[![SERVERLESS](https://img.shields.io/badge/RUNTIME-ML-red?style=for-the-badge)](https://databricks.com/try-databricks)

A simplified, modular implementation of Media Mix Modeling (MMM) using PyMC and Databricks, designed for production use with clean architecture and comprehensive testing.

## Overview

**Media Mix Modeling (MMM)** is a data-driven methodology that enables companies to identify and measure the impact of their marketing campaigns across multiple channels (TV, social media, search, display, etc.). By analyzing historical data and including external factors like holidays and economic conditions, MMM helps businesses:

- Determine which marketing channels contribute most to strategic KPIs (sales, conversions, etc.)
- Understand the impact of outside factors and avoid over-valuing ad spend alone
- Make better-informed decisions about advertising and marketing budget allocation

**Databricks Lakehouse** provides a unified platform for building scalable MMM solutions with automated data ingestion, powerful ML capabilities, and full data transparency through Unity Catalog.

This is a **simplified and modernized** version of the [original Databricks MMM solution accelerator](https://github.com/databricks-industry-solutions/media-mix-modeling), preserving all core functionality while providing:

1. **Data Generation**: Create synthetic MMM data with configurable channels, adstock effects, and saturation curves
2. **Bayesian Modeling**: Fit MMM models using PyMC with proper uncertainty quantification
3. **Agent Interface**: Forecasting, historical analysis, and budget optimization capabilities
4. **Real Data Support**: Work with real MMM datasets (includes Home & Entertainment example)

### What's New vs Original Repo

- ✅ **UV-based** dependency management (replaces Poetry)
- ✅ **Modular src/** structure with single-responsibility modules
- ✅ **Comprehensive pytest** test suite
- ✅ **Simplified notebooks** at root (01, 02, 03, 04)
- ✅ **Pydantic** for configuration validation
- ✅ **Full Databricks integration** (Delta tables, MLflow, Unity Catalog)
- ✅ **Real data utilities** for production use
- ✅ **Clean documentation** in single README

### Preserved from Original

- ✓ All MMM modeling capabilities
- ✓ Adstock transformations (geometric decay)
- ✓ Saturation curves (Hill, logistic)
- ✓ Channel contribution analysis
- ✓ Budget optimization
- ✓ Real dataset examples
- ✓ Databricks deployment patterns

## Architecture

```
media-mix-modeling/
├── src/                          # Core modules
│   ├── data_generation.py        # Synthetic data generation
│   ├── datasets.py               # Real data utilities
│   ├── model.py                  # PyMC-based MMM
│   ├── agent.py                  # Agent for forecasting & optimization
│   ├── optimizer.py              # Budget optimization
│   └── transforms.py             # Adstock & saturation functions
├── tests/                        # Comprehensive test suite
├── data/                         # Example datasets
│   └── he_mmm_data.csv          # Home & Entertainment real data
├── 01_generate_data.py          # Notebook: Generate synthetic data
├── 02_run_model.py              # Notebook: Fit PyMC model
├── 03_agent.py                  # Notebook: Use agent
├── 04_real_data_example.py      # Notebook: Real data example
├── example_config.yaml           # Example configuration file
└── pyproject.toml               # UV-based dependency management
```

## Installation

This project uses [UV](https://github.com/astral-sh/uv) for dependency management.

### On Databricks

Each notebook includes installation cells:

```python
# Cell 1: Install UV
%pip install uv

# Cell 2: Install package with UV
%sh uv pip install .

# Cell 3: Restart Python
%restart_python
```

### Local Development

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies (prod + dev)
uv pip install -e ".[dev]"

# Or just production dependencies
uv pip install -e .

# Run tests
uv run pytest tests/ -v -m "not slow"
```

## Configuration

All configuration is managed through a single YAML file loaded with MLflow's `ModelConfig`. See `example_config.yaml`:

### Configuration Parameters

**Global Settings:**
- `random_seed`: Random seed for reproducible data generation
- `catalog`, `schema`, `synthetic_data_table`: Unity Catalog location for data storage
- `start_date`, `end_date`: Date range for synthetic data

**Outcome Configuration (`outcome`):**
- `name`: Outcome variable name (e.g., "sales")
- `intercept`: Baseline outcome level before channel effects
- `sigma`: Noise/variance in the outcome variable
- `scale`: Multiplier to scale outcome to realistic values

**Channel Configuration (`media.<channel_name>`):**
- `beta`: Channel contribution coefficient (impact strength)
- `min`, `max`: Spend range for the channel
- `sigma`: Variance in the spend signal (randomness)
- `decay`: Enable adstock/carryover effects (true/false)
- `alpha`: Decay rate for adstock (0-1, higher = longer carryover)
- `saturation`: Enable diminishing returns (true/false)
- `mu`: Saturation parameter (higher = slower saturation)

## Quick Start

> **Note**: This implementation is fully compatible with the original [databricks-industry-solutions/media-mix-modeling](https://github.com/databricks-industry-solutions/media-mix-modeling) repository, providing a simplified and modernized approach with the same core capabilities.

### 1. Generate Synthetic Data

Run the `01_generate_data.py` notebook which loads configuration from YAML:

```python
from src.data_generation import DataGenerator
import mlflow

# Load configuration using MLflow ModelConfig
config = mlflow.models.ModelConfig(development_config="example_config.yaml")
generator = DataGenerator.from_config(config)

# Generate data (uses random_seed from config)
df = generator.generate()

# Save to Delta table (uses catalog/schema/table from config)
generator.save_to_delta(df=df)
```

### 2. Fit MMM Model

Run the `02_run_model.py` notebook or use the module directly:

```python
from src.model import MediaMixModel, MMModelConfig, ChannelSpec
import pandas as pd

# Load data
df = spark.table("main.mmm.synthetic_data").toPandas()
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Configure model
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
    # ... more channels
]

config = MMModelConfig(
    outcome_name="sales",
    outcome_scale=100000,
    channels=channels
)

# Fit model
mmm = MediaMixModel(config)
idata = mmm.fit(
    df=df,
    draws=2000,
    tune=1000,
    chains=4
)

# Save to MLflow
mmm.save_to_mlflow(
    experiment_name="/mmm/experiments",
    run_name="baseline_v1"
)
```

### 3. Use Agent for Forecasting & Optimization

Run the `03_agent.py` notebook or use the module directly:

### 4. Work with Real Data

The package includes utilities for working with real MMM datasets. See `04_real_data_example.py` for a complete example:

```python
from src.datasets import load_he_mmm_dataset, prepare_mmm_data, summarize_dataset

# Load real Home & Entertainment dataset
df = load_he_mmm_dataset()

# Get summary
summary = summarize_dataset(df)
print(f"Date range: {summary['date_range']}")
print(f"Media channels: {summary['media_channels']}")

# Prepare for modeling
df_model = prepare_mmm_data(
    df,
    outcome_col='sales',
    media_prefix='mdsp',  # Media spend
    include_controls=['hldy_', 'seas_']  # Add holidays and seasonality
)

# Fit model as usual
mmm = MediaMixModel(config)
idata = mmm.fit(df_model)
```

## Agent Usage

```python
from src.agent import MMAgent, ForecastRequest
from src.optimizer import BudgetConstraints

# Initialize agent
agent = MMAgent(model=mmm, data=df)

# 1. Historical Analysis
results = agent.analyze_historical()
print(f"Model WAIC: {results.model_fit_metrics['waic']}")

# Get channel insights
insights = agent.get_channel_insights()
for channel, insight in insights.items():
    print(insight)

# 2. Forecasting
forecast_request = ForecastRequest(
    future_spend={
        "tv": [35000] * 30,
        "digital": [20000] * 30,
    }
)
forecast = agent.forecast(forecast_request)
print(f"Predicted sales: {forecast.predictions}")

# 3. Budget Optimization
constraints = BudgetConstraints(
    total_budget=400000,
    min_spend_per_channel={"tv": 20000, "digital": 10000},
    max_spend_per_channel={"tv": 150000, "digital": 100000}
)
optimization = agent.optimize_budget(constraints)
print(f"Optimal allocation: {optimization.optimal_allocation}")
print(f"Expected outcome: ${optimization.expected_outcome:,.0f}")
```

## Key Features

### Data Generation

- **Configurable channels** with individual parameters
- **Adstock effects** (geometric decay for carryover)
- **Saturation curves** (logistic saturation for diminishing returns)
- **Delta table integration** for Databricks
- **Reproducible** synthetic data for testing

### Bayesian Modeling

- **PyMC implementation** with proper priors
- **Flexible transformations** (adstock, saturation)
- **Full posterior inference** with MCMC
- **Model diagnostics** (WAIC, LOO, R-hat)
- **MLflow integration** for experiment tracking
- **Channel contributions** calculation

### Agent Capabilities

- **Forecasting** with credible intervals
- **Historical analysis** with performance metrics
- **Budget optimization** using scipy
- **ROAS calculation** per channel
- **Recommendations** generation
- **Extensible** for LangChain/LangGraph integration

## Testing

Comprehensive test suite using pytest:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run only fast tests (skip slow MCMC tests)
uv run pytest -m "not slow"

# Run specific test file
uv run pytest tests/test_data_generation.py
```

## Dependencies

### Production (Databricks Deployment)
- `pymc>=5.16.0` - Bayesian modeling
- `pandas`, `numpy`, `scipy` - Data processing
- `arviz>=0.20.0` - Inference diagnostics
- `pydantic>=2.11.1` - Configuration management
- `databricks-connect>=15.0.0` - Databricks integration
- `mlflow>=2.20.0` - Model tracking
- `delta-spark>=3.0.0` - Delta table support

### Development
- `pytest>=8.3.5` - Testing framework
- `black>=25.1.0` - Code formatting
- `ruff>=0.8.0` - Linting

### Agent (Optional)
- `langgraph>=0.3.22` - Agent framework
- `langchain-core>=0.3.0` - LLM integration
- `databricks-langchain>=0.4.1` - Databricks LLM
- `openai>=1.70.0` - OpenAI integration
- `plotly>=6.0.1` - Visualization

## Project Structure

### Modular Design

- **Single Responsibility**: Each module has a clear, focused purpose
- **Composition over Inheritance**: Favor functions and composition
- **Testability**: Clean dependency injection for easy testing
- **Databricks-Ready**: Designed for deployment on Databricks

### Module Responsibilities

- `data_generation.py`: Generate synthetic MMM data
- `transforms.py`: Reusable transformation functions
- `model.py`: Core PyMC modeling logic
- `optimizer.py`: Budget optimization algorithms
- `agent.py`: High-level interface for common tasks

## Usage on Databricks

### Setup

1. Clone this repo to Databricks Repos
2. Create a cluster with ML runtime (13.3 LTS or higher)
3. Install the package: `%pip install -e .`

### Workflow

1. **Generate Data**: Run `01_generate_data.py`
   - Creates synthetic data
   - Saves to Unity Catalog Delta table

2. **Fit Model**: Run `02_run_model.py`
   - Loads data from Delta table
   - Fits Bayesian MMM with PyMC
   - Logs to MLflow
   - Saves inference data

3. **Use Agent**: Run `03_agent.py`
   - Loads fitted model
   - Performs forecasting
   - Analyzes historical performance
   - Optimizes budget allocation
   - Saves results to Delta table

4. **Real Data Example**: Run `04_real_data_example.py`
   - Loads real Home & Entertainment dataset
   - Prepares data for modeling
   - Fits model with real data
   - Demonstrates production workflow

### Unity Catalog Integration

All data is stored in Unity Catalog:

```
main                          # Catalog
└── mmm                       # Schema
    ├── synthetic_data        # Generated data
    ├── optimization_results  # Agent outputs
    └── ...
```

## Best Practices

### Model Configuration

- **Start simple**: Begin with basic priors, add complexity as needed
- **Check diagnostics**: Always review R-hat, ESS, and trace plots
- **Prior sensitivity**: Test different priors to ensure robustness
- **Scaling**: Properly scale your outcome variable for better sampling

### Optimization

- **Realistic constraints**: Set min/max spend based on business rules
- **Multiple scenarios**: Compare different budget levels
- **Validation**: Back-test optimized allocations on holdout data

### Production Deployment

- **Version control**: Track model versions in MLflow
- **Monitoring**: Set up alerts for unusual ROAS changes
- **Retraining**: Schedule regular model updates with new data
- **Documentation**: Keep configuration and business logic documented

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Authors

- Scott McKean (scott.mckean@databricks.com)
- Corey Abshire (corey.abshire@databricks.com)
- Layla El-Sayed (layla@databricks.com)

## Support

This project is provided AS-IS for exploration purposes. For issues, please file a GitHub issue.

---

&copy; 2025 Databricks, Inc. All rights reserved.

| Library | Description | License | Source |
|---------|-------------|---------|--------|
| pytest | Python testing framework | MIT | https://github.com/pytest-dev/pytest |
| pymc | Probabilistic Programming in Python | Apache 2.0 | https://github.com/pymc-devs/pymc |
| pydantic | Data validation using Python type hints | MIT | https://github.com/pydantic/pydantic |
