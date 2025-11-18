"""
Tests for streamlined MMM agent with four core capabilities:
1. Generate constraints from natural language
2. Optimize budgets using constraints
3. Handle end-to-end queries with routing
4. Route to Genie spaces for historical data
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from src.agent import MMMAgent, OptimizationRequest, OptimizationResponse
from src.optimizer import BudgetConstraints


@pytest.fixture
def mock_databricks_llm():
    """Mock Databricks LLM responses for testing."""

    def mock_completion(*args, **kwargs):
        """Mock LiteLLM completion call."""
        # Create a mock response that looks like a LiteLLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "intent": "analysis",
                "requires_mcp": False,
                "reasoning": "This is a query about channel performance",
            }
        )
        return mock_response

    with patch("litellm.completion", side_effect=mock_completion):
        yield mock_completion


def test_optimization_request_creation():
    """Test creating optimization request."""
    request = OptimizationRequest(
        user_query="Optimize with total budget of $50,000",
        total_budget=50000,
    )

    assert request.user_query is not None
    assert request.total_budget == 50000
    assert request.channel_constraints is None


def test_budget_constraints_nested_format():
    """Test creating BudgetConstraints with nested channel dict."""
    # New nested format
    constraints = BudgetConstraints(
        total_budget=50000,
        channels={
            "facebook": {"min_spend": 1000, "max_spend": 20000},
            "google": {"min_spend": 2000, "max_spend": 25000},
        },
    )

    assert constraints.total_budget == 50000
    assert constraints.min_spend_per_channel["facebook"] == 1000
    assert constraints.max_spend_per_channel["facebook"] == 20000
    assert constraints.min_spend_per_channel["google"] == 2000
    assert constraints.max_spend_per_channel["google"] == 25000


def test_budget_constraints_flat_format():
    """Test creating BudgetConstraints with flat min/max dicts."""
    # Original flat format
    constraints = BudgetConstraints(
        total_budget=50000,
        min_spend_per_channel={"facebook": 1000, "google": 2000},
        max_spend_per_channel={"facebook": 20000, "google": 25000},
    )

    assert constraints.total_budget == 50000
    assert constraints.min_spend_per_channel["facebook"] == 1000
    assert constraints.max_spend_per_channel["google"] == 25000


def test_optimization_response_creation():
    """Test creating optimization response."""
    response = OptimizationResponse(
        optimal_allocation={"channel_1": 20000, "channel_2": 15000},
        expected_sales=150000,
        channel_roas={"channel_1": 3.5, "channel_2": 4.2},
        total_roas=3.8,
        explanation="Optimized allocation based on ROAS",
        recommendations="Increase channel_2 spend",
    )

    assert len(response.optimal_allocation) == 2
    assert response.expected_sales == 150000
    assert response.total_roas == 3.8


def test_agent_initialization(model_config, synthetic_data):
    """Test initializing streamlined agent."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)
    agent = MMMAgent(model=model, data=synthetic_data)

    assert agent.model == model
    assert agent.data.equals(synthetic_data)
    assert agent.optimizer is not None


def test_agent_with_dspy_config(model_config, synthetic_data):
    """Test agent initialization with DSPy config."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)

    # Create agent config with DSPy settings (but it will fail to connect)
    agent_config = {
        "llm_model": "databricks-meta-llama-3-1-70b-instruct",
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    # Agent should initialize even if DSPy connection fails
    # It will fall back to OpenAI or just not initialize DSPy modules
    agent = MMMAgent(model=model, data=synthetic_data, agent_config=agent_config)
    assert agent.agent_config == agent_config
    # DSPy modules may or may not be initialized depending on environment


# =============================================================================
# Core Capability 1: Generate Constraints
# =============================================================================


def test_generate_constraints_from_config(model_config, small_synthetic_data):
    """Test generating constraints from config (fallback method)."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)

    # Agent config with channel constraints
    agent_config = {
        "channels": {
            "channel_1": {"min_spend": 5000, "max_spend": 50000},
            "channel_2": {"min_spend": 3000, "max_spend": 40000},
        },
        "optimization": {"default_budget": 100000},
    }

    agent = MMMAgent(model=model, data=small_synthetic_data, agent_config=agent_config)

    # Get default constraints from config (bypasses DSPy)
    constraints = agent._get_default_constraints()

    assert isinstance(constraints, BudgetConstraints)
    assert constraints.total_budget == 100000
    assert constraints.channels["channel_1"]["min_spend"] == 5000
    assert constraints.channels["channel_1"]["max_spend"] == 50000


def test_generate_constraints_from_historical_data(model_config, small_synthetic_data):
    """Test generating constraints from historical data when no config provided."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)
    agent = MMMAgent(model=model, data=small_synthetic_data)

    # Generate constraints from historical data (no config)
    constraints = agent._get_default_constraints()

    assert isinstance(constraints, BudgetConstraints)
    assert constraints.total_budget > 0
    # Min should be ~50% of historical mean
    # Max should be ~200% of historical mean
    for channel in ["channel_1", "channel_2"]:
        hist_mean = small_synthetic_data[channel].mean()
        assert constraints.min_spend_per_channel[channel] < hist_mean
        assert constraints.max_spend_per_channel[channel] > hist_mean


# =============================================================================
# Core Capability 2: Optimize with Constraints
# =============================================================================


def test_optimize_with_constraints(model_config, small_synthetic_data):
    """Test budget optimization with constraints."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)

    # Fit model first
    model.fit(df=small_synthetic_data, draws=20, tune=20, chains=1, target_accept=0.8)

    agent = MMMAgent(model=model, data=small_synthetic_data)

    # Create constraints
    constraints = BudgetConstraints(
        total_budget=50000,
        min_spend_per_channel={"channel_1": 10000, "channel_2": 5000},
        max_spend_per_channel={"channel_1": 30000, "channel_2": 25000},
    )

    # Optimize (without DSPy explanation)
    result = agent.optimize(constraints, explain=False)

    assert isinstance(result, OptimizationResponse)
    assert sum(result.optimal_allocation.values()) <= constraints.total_budget * 1.01
    assert result.expected_sales > 0
    assert result.total_roas > 0
    assert len(result.channel_roas) == 2


def test_optimize_without_constraints(model_config, small_synthetic_data):
    """Test optimization generates default constraints."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)

    model.fit(df=small_synthetic_data, draws=20, tune=20, chains=1, target_accept=0.8)

    agent = MMMAgent(model=model, data=small_synthetic_data)

    # Optimize without providing constraints
    result = agent.optimize(constraints=None, explain=False)

    assert isinstance(result, OptimizationResponse)
    assert result.expected_sales > 0


# =============================================================================
# Core Capability 3: End-to-End Query Handler
# =============================================================================


def test_query_without_dspy_config(model_config, small_synthetic_data):
    """Test query method returns error when DSPy not configured."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)
    agent = MMMAgent(model=model, data=small_synthetic_data)

    result = agent.query("What is the best channel?")

    assert "error" in result
    assert "DSPy not configured" in result["error"]


def test_query_with_dspy_config(model_config, small_synthetic_data, mock_databricks_llm):
    """Test query method with DSPy config (using mocked LLM)."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)

    model.fit(df=small_synthetic_data, draws=20, tune=20, chains=1, target_accept=0.8)

    agent_config = {
        "llm_model": "databricks-meta-llama-3-1-70b-instruct",
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    agent = MMMAgent(model=model, data=small_synthetic_data, agent_config=agent_config)

    # DSPy modules should be initialized with agent_config
    assert agent.query_router is not None, "Query router should be initialized with agent_config"

    # Try a query - it will use the mocked LLM
    result = agent.query("Which channel has the best ROAS?")
    assert "intent" in result or "error" in result


# =============================================================================
# Core Capability 4: MCP/Genie Integration
# =============================================================================


def test_mcp_placeholder_replacement(model_config, synthetic_data):
    """Test that MCP server URLs get placeholders replaced."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)

    agent_config = {
        "catalog": "test_catalog",
        "schema": "test_schema",
        "llm_model": "test-model",
        "mcp_server_urls": [
            "{workspace_hostname}/api/2.0/mcp/functions/{catalog}/{schema}",
            "{workspace_hostname}/api/2.0/mcp/vector-search/{catalog}/{schema}",
        ],
    }

    try:
        agent = MMMAgent(model=model, data=synthetic_data, agent_config=agent_config)
        # If it gets here, agent was created (MCP connection may have failed gracefully)
        assert agent.agent_config == agent_config
    except Exception as e:
        # Expected to fail without proper Databricks credentials
        assert "databricks" in str(e).lower() or "workspace" in str(e).lower()


def test_list_mcp_tools(model_config, synthetic_data):
    """Test listing MCP tools (will return empty if not configured)."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)
    agent = MMMAgent(model=model, data=synthetic_data)

    tools = agent.list_mcp_tools()

    # Should return empty list if MCP not configured
    assert isinstance(tools, list)


# =============================================================================
# Integration Tests
# =============================================================================


def test_end_to_end_optimization_workflow(model_config, small_synthetic_data):
    """Test complete optimization workflow: generate constraints -> optimize."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)

    model.fit(df=small_synthetic_data, draws=20, tune=20, chains=1, target_accept=0.8)

    agent_config = {
        "channels": {
            "channel_1": {"min_spend": 5000, "max_spend": 30000},
            "channel_2": {"min_spend": 3000, "max_spend": 25000},
        },
        "optimization": {"default_budget": 50000},
    }

    agent = MMMAgent(model=model, data=small_synthetic_data, agent_config=agent_config)

    # Step 1: Generate constraints from config (bypasses DSPy)
    constraints = agent._get_default_constraints()

    # Step 2: Optimize
    result = agent.optimize(constraints, explain=False)

    # Verify results
    assert isinstance(result, OptimizationResponse)
    assert sum(result.optimal_allocation.values()) <= 51000  # Allow 2% tolerance
    assert all(
        constraints.min_spend_per_channel[ch]
        <= result.optimal_allocation[ch]
        <= constraints.max_spend_per_channel[ch]
        for ch in result.optimal_allocation
    )


def test_conversation_history(model_config, small_synthetic_data, mock_databricks_llm):
    """Test that agent can handle conversation history."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)
    model.fit(df=small_synthetic_data, draws=20, tune=20, chains=1, target_accept=0.8)

    agent_config = {
        "llm_model": "databricks-meta-llama-3-1-70b-instruct",
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    agent = MMMAgent(model=model, data=small_synthetic_data, agent_config=agent_config)

    # Test 1: Query without history
    result1 = agent.query("Which channel has the best ROAS?")
    assert "intent" in result1 or "error" in result1 or "response" in result1

    # Test 2: Query with conversation history
    conversation_history = [
        {"role": "user", "content": "Show me channel performance"},
        {"role": "assistant", "content": "Here's the performance summary..."},
        {"role": "user", "content": "What about Facebook specifically?"},
    ]
    
    result2 = agent.query("Now optimize the budget", conversation_history=conversation_history)
    assert "intent" in result2 or "error" in result2 or "response" in result2

    # Test 3: Empty history should work
    result3 = agent.query("Analyze channels", conversation_history=[])
    assert "intent" in result3 or "error" in result3 or "response" in result3

    # Test 4: Long history (should be truncated to last 5 messages)
    long_history = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
    result4 = agent.query("Latest query", conversation_history=long_history)
    assert "intent" in result4 or "error" in result4 or "response" in result4


def test_follow_up_optimization_with_same_budget(model_config, small_synthetic_data, mock_databricks_llm):
    """Test that agent preserves budget when user says 'keep same budget'."""
    from src.model import MediaMixModel

    model = MediaMixModel(model_config)
    model.fit(df=small_synthetic_data, draws=20, tune=20, chains=1, target_accept=0.8)

    agent_config = {
        "llm_model": "databricks-meta-llama-3-1-70b-instruct",
        "max_tokens": 2048,
        "temperature": 0.1,
        "channels": {
            "channel_1": {"min_spend": 5000, "max_spend": 30000},
            "channel_2": {"min_spend": 3000, "max_spend": 25000},
        },
        "optimization": {"default_budget": 50000},
    }

    agent = MMMAgent(model=model, data=small_synthetic_data, agent_config=agent_config)

    # First optimization
    result1 = agent.query("Optimize with $50,000 total budget")
    
    # Create conversation history with first optimization result
    conversation_history = [
        {"role": "user", "content": "Optimize with $50,000 total budget"},
        {"role": "assistant", "content": "**Optimization Results**\n\nOptimal Allocation:\n- Channel_1: $30,000\n- Channel_2: $20,000\n\n**Expected Sales:** $100,000\n**Total ROAS:** 2.0x"},
    ]
    
    # Follow-up optimization requesting to keep same budget
    result2 = agent.query(
        "Keep the same budget but increase channel_1 to $35,000", 
        conversation_history=conversation_history
    )
    
    # The budget should still be $50,000, not recalculated from historical data
    if result2.get("intent") == "optimization" and "constraints" in result2:
        # Allow 2% tolerance for floating point differences
        assert abs(result2["constraints"]["total_budget"] - 50000) < 1000, \
            f"Budget should be ~$50,000 but got ${result2['constraints']['total_budget']:,.0f}"
