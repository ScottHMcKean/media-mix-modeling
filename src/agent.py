"""
Streamlined MMM Agent for budget optimization and analysis.

This module provides a focused agent interface powered by DSPy with four core capabilities:
1. Generate budget constraints from natural language
2. Optimize budgets using constraints
3. Handle end-to-end user queries with intelligent routing
4. Route to Genie spaces via MCP for historical data queries
"""

from typing import Dict, List, Optional, Any
import json

import dspy
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.model import MediaMixModel
from src.optimizer import BudgetConstraints, BudgetOptimizer, OptimizationResult


# =============================================================================
# Pydantic Models
# =============================================================================


class OptimizationRequest(BaseModel):
    """Request for budget optimization."""

    user_query: str = Field(description="Natural language optimization request")
    total_budget: Optional[float] = Field(default=None, description="Total budget constraint")
    channel_constraints: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None, description="Optional channel-specific constraints"
    )


class OptimizationResponse(BaseModel):
    """Response from budget optimization."""

    optimal_allocation: Dict[str, float] = Field(description="Optimal spend per channel")
    expected_sales: float = Field(description="Expected total sales")
    channel_roas: Dict[str, float] = Field(description="ROAS by channel")
    total_roas: float = Field(description="Overall portfolio ROAS")
    explanation: str = Field(description="Natural language explanation")
    recommendations: str = Field(description="Actionable recommendations")


# =============================================================================
# DSPy Signatures
# =============================================================================


class ConstraintGenerationSignature(dspy.Signature):
    """Extract budget constraints from natural language request."""

    user_request: str = dspy.InputField(desc="User's natural language optimization request")
    historical_spend: str = dspy.InputField(desc="Historical spend statistics by channel")
    channel_performance: str = dspy.InputField(desc="Current ROAS and performance metrics")

    total_budget: float = dspy.OutputField(desc="Total budget for optimization")
    min_spend_per_channel: str = dspy.OutputField(
        desc="JSON dict mapping channel names to minimum spend amounts"
    )
    max_spend_per_channel: str = dspy.OutputField(
        desc="JSON dict mapping channel names to maximum spend amounts"
    )
    reasoning: str = dspy.OutputField(desc="Explanation of constraint choices")


class OptimizationExplanationSignature(dspy.Signature):
    """Explain optimization results in natural language."""

    user_request: str = dspy.InputField(desc="Original user request")
    current_allocation: str = dspy.InputField(desc="Current/historical allocation")
    optimal_allocation: str = dspy.InputField(desc="Optimized allocation")
    performance_metrics: str = dspy.InputField(desc="ROAS and sales metrics")

    explanation: str = dspy.OutputField(desc="Clear explanation of optimization results")
    recommendations: str = dspy.OutputField(desc="Actionable next steps")
    key_changes: str = dspy.OutputField(desc="Highlight of major allocation changes")


class QueryRoutingSignature(dspy.Signature):
    """Route user queries to appropriate handler."""

    user_query: str = dspy.InputField(desc="User's question or request")
    available_capabilities: str = dspy.InputField(
        desc="List of available capabilities (optimization, historical_data, analysis)"
    )

    intent: str = dspy.OutputField(
        desc="Primary intent: 'optimization', 'historical_data', 'analysis', or 'general'"
    )
    requires_mcp: bool = dspy.OutputField(
        desc="Whether query requires MCP access to Genie/Unity Catalog"
    )
    reasoning: str = dspy.OutputField(desc="Why this routing was chosen")


class HistoricalDataQuerySignature(dspy.Signature):
    """Generate SQL or Genie query for historical data requests."""

    user_query: str = dspy.InputField(desc="User's question about historical data")
    available_tables: str = dspy.InputField(desc="Available tables and their schemas")

    genie_query: str = dspy.OutputField(desc="Natural language query for Genie space")
    expected_insights: str = dspy.OutputField(desc="What insights this query will provide")


class HistoricalGenieSignature(dspy.Signature):
    """Use Genie space tools to answer historical data questions."""

    user_request: str = dspy.InputField(desc="User's question about historical data")
    process_result: str = dspy.OutputField(desc="Processed result from Genie query")


# =============================================================================
# DSPy Modules
# =============================================================================


class ConstraintGenerator(dspy.Module):
    """Generate budget constraints from natural language."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ConstraintGenerationSignature)

    def forward(self, user_request: str, historical_spend: str, channel_performance: str):
        return self.generate(
            user_request=user_request,
            historical_spend=historical_spend,
            channel_performance=channel_performance,
        )


class OptimizationExplainer(dspy.Module):
    """Explain optimization results."""

    def __init__(self):
        super().__init__()
        self.explain = dspy.ChainOfThought(OptimizationExplanationSignature)

    def forward(
        self,
        user_request: str,
        current_allocation: str,
        optimal_allocation: str,
        performance_metrics: str,
    ):
        return self.explain(
            user_request=user_request,
            current_allocation=current_allocation,
            optimal_allocation=optimal_allocation,
            performance_metrics=performance_metrics,
        )


class QueryRouter(dspy.Module):
    """Route queries to appropriate handlers."""

    def __init__(self):
        super().__init__()
        self.route = dspy.ChainOfThought(QueryRoutingSignature)

    def forward(self, user_query: str, available_capabilities: str):
        return self.route(user_query=user_query, available_capabilities=available_capabilities)


class HistoricalDataHandler(dspy.Module):
    """Handle historical data queries via Genie."""

    def __init__(self):
        super().__init__()
        self.query = dspy.ChainOfThought(HistoricalDataQuerySignature)

    def forward(self, user_query: str, available_tables: str):
        return self.query(user_query=user_query, available_tables=available_tables)


class HistoricalGenie(dspy.Module):
    """Handle historical data queries via fastmcp Genie integration."""

    def __init__(self, tools: List = None):
        super().__init__()
        self.tools = tools or []
        if tools:
            # Use ReAct with Genie tools
            self.react = dspy.ReAct(HistoricalGenieSignature, tools=self.tools)
        else:
            # Fallback to simple ChainOfThought
            self.react = dspy.ChainOfThought(HistoricalGenieSignature)

    def forward(self, user_request: str):
        return self.react(user_request=user_request)


# =============================================================================
# Main MMM Agent
# =============================================================================


class MMMAgent:
    """
    Streamlined MMM Agent with four core capabilities:

    1. Generate constraints from natural language
    2. Optimize budgets using constraints
    3. Handle end-to-end queries with routing
    4. Route to Genie spaces for historical data
    """

    def __init__(
        self, model: MediaMixModel, data: pd.DataFrame, agent_config: Optional[dict] = None
    ):
        """
        Initialize the MMM Agent.

        Args:
            model: Fitted MediaMixModel instance
            data: Historical data used for model fitting
            agent_config: Configuration dict with LLM settings and MCP servers
        """
        self.model = model
        self.data = data
        self.agent_config = agent_config or {}
        self.optimizer = BudgetOptimizer(model, data)

        # MCP client for Genie/Unity Catalog access
        self.mcp_client = None
        self.workspace_client = None

        # fastmcp client for Historical Genie
        self.genie_client = None
        self.genie_tools = []

        # Initialize DSPy modules
        if agent_config:
            self._configure_dspy(agent_config)
            self.constraint_generator = ConstraintGenerator()
            self.optimization_explainer = OptimizationExplainer()
            self.query_router = QueryRouter()
            self.historical_data_handler = HistoricalDataHandler()
            self.historical_genie = None  # Initialized async when needed
        else:
            self.constraint_generator = None
            self.optimization_explainer = None
            self.query_router = None
            self.historical_data_handler = None
            self.historical_genie = None

    def _configure_dspy(self, agent_config: dict):
        """Configure DSPy with Databricks."""
        from databricks.sdk import WorkspaceClient

        # Initialize WorkspaceClient (uses environment for auth)
        self.workspace_client = WorkspaceClient()
        workspace_hostname = self.workspace_client.config.host

        # Configure DSPy with Databricks serving endpoint
        model_name = agent_config.get("llm_model", "databricks-meta-llama-3-1-70b-instruct")

        # Configure DSPy LM with databricks/ prefix for LiteLLM routing
        lm = dspy.LM(
            model=f"databricks/{model_name}",
            max_tokens=agent_config.get("max_tokens", 2048),
            temperature=agent_config.get("temperature", 0.1),
        )
        dspy.configure(lm=lm)

        # Initialize MCP client for Unity Catalog functions
        self._setup_mcp_client(workspace_hostname, agent_config)

        print(f"✓ DSPy configured with Databricks endpoint: {model_name}")
        print(f"✓ Workspace: {workspace_hostname}")

    def _setup_mcp_client(self, workspace_hostname: str, agent_config: dict):
        """Setup MCP client for Unity Catalog functions or Genie spaces.

        Follows the pattern from:
        https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp

        Supports three server types:
        - system_ai: System tools (python_exec, sql_query, etc.)
        - catalog: Unity Catalog functions
        - genie: Genie space for natural language queries
        """
        from databricks_mcp import DatabricksMCPClient

        # Get MCP configuration
        mcp_config = agent_config.get("mcp", {})
        server_type = mcp_config.get("server_type", "system_ai")

        # Construct server URL based on server type
        if server_type == "genie":
            # Genie Space endpoint for natural language queries
            genie_space_id = mcp_config.get("genie_space_id")
            if not genie_space_id or genie_space_id == "YOUR_GENIE_SPACE_ID":
                print("⚠️  Warning: Genie space ID not configured. Falling back to system AI tools.")
                print("   Set mcp.genie_space_id in your config to enable Genie queries.")
                server_url = f"{workspace_hostname}/api/2.0/mcp/functions/system/ai"
            else:
                server_url = f"{workspace_hostname}/api/2.0/mcp/genie/{genie_space_id}"
        elif server_type == "catalog":
            # Catalog/schema specific functions
            catalog = mcp_config.get("catalog", agent_config.get("catalog", "main"))
            schema = mcp_config.get("schema", agent_config.get("schema", "mmm"))
            server_url = f"{workspace_hostname}/api/2.0/mcp/functions/{catalog}/{schema}"
        else:  # system_ai (default)
            # System AI tools (python_exec, sql_query, vector_search)
            server_url = f"{workspace_hostname}/api/2.0/mcp/functions/system/ai"

        # Initialize MCP client with server_url (singular!)
        # Per Databricks docs, use server_url not server_urls
        self.mcp_client = DatabricksMCPClient(
            server_url=server_url, workspace_client=self.workspace_client
        )

        print(f"✓ MCP client configured ({server_type}): {server_url}")

    async def _initialize_historical_genie(self):
        """Initialize the Historical Genie fastmcp client."""
        if self.historical_genie is not None:
            # Already initialized
            return

        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport
        from databricks.sdk.core import Config
        import nest_asyncio

        # Apply nest_asyncio to allow nested event loops (needed for notebooks)
        nest_asyncio.apply()

        genie_config = self.agent_config.get("historical_genie", {})

        if not genie_config.get("enabled", False):
            print("Historical Genie not enabled in config")
            return

        genie_space_id = genie_config.get("genie_space_id")
        if not genie_space_id or genie_space_id == "YOUR_GENIE_SPACE_ID":
            print("Warning: genie_space_id not configured. Set it in agent config.")
            return

        # Get hostname from workspace client
        hostname = genie_config.get("hostname") or self.workspace_client.config.host

        # Get auth token
        config = Config()
        token = config.oauth_token().access_token

        # Create transport
        transport = StreamableHttpTransport(
            url=f"{hostname}/api/2.0/mcp/genie/{genie_space_id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Create client
        self.genie_client = Client[StreamableHttpTransport](transport)

        # Initialize tools within the client context
        async with self.genie_client:
            tools = await self.genie_client.list_tools()
            print(f"✓ Historical Genie connected - found {len(tools)} tools")

            # Convert to DSPy tools
            self.genie_tools = []
            for tool in tools:
                dspy_tool = dspy.Tool.from_mcp_tool(self.genie_client, tool)
                self.genie_tools.append(dspy_tool)

            # Initialize the HistoricalGenie module
            self.historical_genie = HistoricalGenie(tools=self.genie_tools)
            print(f"✓ Historical Genie module initialized with {len(self.genie_tools)} tools")

    # =========================================================================
    # Core Capability 1: Generate Constraints
    # =========================================================================

    def generate_constraints(self, user_request: str) -> BudgetConstraints:
        """
        Generate budget constraints from natural language request using DSPy.

        Args:
            user_request: Natural language description of constraints
                         e.g., "Optimize with max Facebook spend of $20k and total budget $50k"

        Returns:
            BudgetConstraints object ready for optimization

        Example:
            >>> constraints = agent.generate_constraints(
            ...     "Keep Facebook under $20k, LinkedIn at least $5k, total budget $50k"
            ... )
        """
        return self._generate_constraints_with_dspy(user_request)

    def _generate_constraints_with_dspy(self, user_request: str) -> BudgetConstraints:
        """Use DSPy to extract constraints from natural language."""
        # Get historical spend statistics
        channel_names = [c.name for c in self.model.config.channels]
        historical_spend = {}
        for channel in channel_names:
            if channel in self.data.columns:
                historical_spend[channel] = {
                    "mean": float(self.data[channel].mean()),
                    "min": float(self.data[channel].min()),
                    "max": float(self.data[channel].max()),
                }

        # Get channel performance
        performance = self.model.get_channel_performance_summary(self.data)
        channel_performance = performance[["total_spend", "roas"]].to_dict("index")

        # Call DSPy to generate constraints
        result = self.constraint_generator(
            user_request=user_request,
            historical_spend=json.dumps(historical_spend, indent=2),
            channel_performance=json.dumps(channel_performance, indent=2),
        )

        # Parse the response
        try:
            total_budget = float(result.total_budget)
            min_spend_dict = json.loads(result.min_spend_per_channel)
            max_spend_dict = json.loads(result.max_spend_per_channel)

            # Build nested channels dict for cleaner API
            channels = {}
            for ch in channel_names:
                channels[ch] = {
                    "min_spend": min_spend_dict.get(ch, 0),
                    "max_spend": max_spend_dict.get(ch, total_budget),
                }

            print(f"\n✓ Constraints generated from request")
            print(f"Total Budget: ${total_budget:,.0f}")
            print(f"Channels: {json.dumps(channels, indent=2)}")
            print(f"Reasoning: {result.reasoning}")

            return BudgetConstraints(total_budget=total_budget, channels=channels)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse DSPy constraints: {e}") from e

    def _get_default_constraints(self) -> BudgetConstraints:
        """Generate sensible default constraints from config and historical data."""
        channel_names = [c.name for c in self.model.config.channels]

        # Use config if available
        if self.agent_config and "channels" in self.agent_config:
            channel_config = self.agent_config["channels"]
            channels = {}
            for ch in channel_names:
                ch_config = channel_config.get(ch, {})
                channels[ch] = {
                    "min_spend": ch_config.get("min_spend", 0),
                    "max_spend": ch_config.get("max_spend", 1e6),
                }

            # Calculate total budget from min spends if not specified
            min_total = sum(c["min_spend"] for c in channels.values())
            total_budget = self.agent_config.get("optimization", {}).get(
                "default_budget", min_total * 2
            )
        else:
            # Use historical data as baseline
            channels = {}
            for channel in channel_names:
                if channel in self.data.columns:
                    hist_mean = self.data[channel].mean()
                    channels[channel] = {
                        "min_spend": hist_mean * 0.5,
                        "max_spend": hist_mean * 2.0,
                    }
                else:
                    channels[channel] = {"min_spend": 0, "max_spend": 100000}

            total_budget = sum(
                self.data[ch].mean() for ch in channel_names if ch in self.data.columns
            )

        return BudgetConstraints(total_budget=total_budget, channels=channels)

    # =========================================================================
    # Core Capability 2: Optimize with Constraints
    # =========================================================================

    def optimize(
        self, constraints: Optional[BudgetConstraints] = None, explain: bool = True
    ) -> OptimizationResponse:
        """
        Optimize budget allocation using constraints.

        Args:
            constraints: Budget constraints (if None, uses defaults from config)
            explain: Whether to generate natural language explanation

        Returns:
            OptimizationResponse with allocation and explanation

        Example:
            >>> constraints = agent.generate_constraints("Total budget $50k")
            >>> result = agent.optimize(constraints)
            >>> print(result.explanation)
        """
        # Generate constraints if not provided
        if constraints is None:
            constraints = self._get_default_constraints()

        # Run optimization
        opt_result = self.optimizer.optimize(constraints)

        # Calculate ROAS for each channel
        channel_roas = {}
        for channel, spend in opt_result.optimal_allocation.items():
            contrib = opt_result.channel_contributions.get(channel, 0)
            channel_roas[channel] = contrib / spend if spend > 0 else 0

        # Calculate total ROAS
        total_spend = sum(opt_result.optimal_allocation.values())
        total_sales = sum(opt_result.channel_contributions.values())
        total_roas = total_sales / total_spend if total_spend > 0 else 0

        # Generate explanation if requested
        if explain and self.optimization_explainer:
            explanation, recommendations = self._explain_optimization(
                constraints, opt_result, channel_roas
            )
        else:
            # Convert list of recommendations to string
            explanation = (
                "\n".join(opt_result.recommendations)
                if isinstance(opt_result.recommendations, list)
                else opt_result.recommendations
            )
            recommendations = "See optimal allocation above."

        return OptimizationResponse(
            optimal_allocation=opt_result.optimal_allocation,
            expected_sales=total_sales,
            channel_roas=channel_roas,
            total_roas=total_roas,
            explanation=explanation,
            recommendations=recommendations,
        )

    def _explain_optimization(
        self,
        constraints: BudgetConstraints,
        opt_result: OptimizationResult,
        channel_roas: Dict[str, float],
    ) -> tuple[str, str]:
        """Generate natural language explanation of optimization."""
        # Get current allocation from historical data
        channel_names = [c.name for c in self.model.config.channels]
        current_allocation = {
            ch: float(self.data[ch].mean()) for ch in channel_names if ch in self.data.columns
        }

        # Format for DSPy
        performance_metrics = {
            "channel_roas": channel_roas,
            "total_roas": sum(opt_result.channel_contributions.values())
            / sum(opt_result.optimal_allocation.values()),
            "expected_sales": sum(opt_result.channel_contributions.values()),
        }

        result = self.optimization_explainer(
            user_request=f"Optimize budget with total of ${constraints.total_budget:,.0f}",
            current_allocation=json.dumps(current_allocation, indent=2),
            optimal_allocation=json.dumps(opt_result.optimal_allocation, indent=2),
            performance_metrics=json.dumps(performance_metrics, indent=2),
        )

        return result.explanation, result.recommendations

    # =========================================================================
    # Core Capability 3: End-to-End Query Handler
    # =========================================================================

    def query(
        self, user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Handle end-to-end user queries with intelligent routing.

        Routes queries to appropriate handlers:
        - Optimization requests → generate_constraints + optimize
        - Historical data queries → MCP/Genie
        - General analysis → model insights

        Args:
            user_query: Natural language query
            conversation_history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]

        Returns:
            Dict with response and metadata

        Example:
            >>> response = agent.query("What if I increase Facebook spend by 20%?")
            >>> response = agent.query("Show me sales data from last quarter")
            >>> response = agent.query("Which channel has the best ROAS?")
            >>> # With conversation history:
            >>> history = [{"role": "user", "content": "Show me Facebook"}, {"role": "assistant", "content": "..."}]
            >>> response = agent.query("Now optimize it", conversation_history=history)
        """
        if not self.query_router:
            return {
                "error": "DSPy not configured. Please provide agent_config.",
                "query": user_query,
            }

        # Build context from conversation history
        context_str = ""
        if conversation_history:
            # Include last 5 messages for context (to avoid token limits)
            recent_history = conversation_history[-5:]
            context_parts = []
            for msg in recent_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context_parts.append(f"{role}: {content[:200]}")  # Limit each message to 200 chars
            context_str = "\n".join(context_parts)

        # Augment query with context if available
        query_with_context = user_query
        if context_str:
            query_with_context = (
                f"Previous conversation:\n{context_str}\n\nCurrent question: {user_query}"
            )

        # Route the query
        capabilities = """
        - optimization: Budget allocation optimization with constraints
        - historical_data: Query historical sales and spend data via Genie/Unity Catalog
        - historical_genie: Query historical data using Genie space (for peak sales, trends, etc.)
        - analysis: Analyze channel performance, ROAS, contributions
        """

        routing = self.query_router(
            user_query=query_with_context, available_capabilities=capabilities
        )

        print(f"\n🔀 Query routed to: {routing.intent}")
        print(f"Reasoning: {routing.reasoning}")

        # Handle based on intent
        if routing.intent == "optimization":
            return self._handle_optimization_query(user_query, conversation_history)
        elif routing.intent == "historical_genie" or (
            routing.intent == "historical_data" and self._is_genie_enabled()
        ):
            # Use Historical Genie for historical data queries
            return self._handle_historical_genie_query(user_query)
        elif routing.intent == "historical_data" and routing.requires_mcp:
            return self._handle_historical_data_query(user_query)
        elif routing.intent == "analysis":
            return self._handle_analysis_query(user_query)
        else:
            return {"response": "I'm not sure how to handle that query.", "intent": routing.intent}

    def _handle_optimization_query(
        self, user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Handle optimization-related queries with conversation context."""
        try:
            # Look for previous optimization results in conversation history
            previous_budget = None
            previous_allocation = None

            if conversation_history:
                # Search backwards for the most recent optimization result
                for msg in reversed(conversation_history):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        # Try to extract budget from previous optimization
                        if "Total Budget:" in content or "total budget" in content.lower():
                            import re

                            # Look for budget amounts like $50,000 or 50000
                            budget_match = re.search(r"\$?([\d,]+)", content)
                            if budget_match:
                                budget_str = budget_match.group(1).replace(",", "")
                                try:
                                    previous_budget = float(budget_str)
                                except ValueError:
                                    pass

                        # Look for allocation info
                        if "Optimal Allocation:" in content or "allocation" in content.lower():
                            # Found a previous optimization
                            if previous_budget:
                                break

            # Augment user query with context if we found previous budget
            enhanced_query = user_query
            if previous_budget and ("same" in user_query.lower() or "keep" in user_query.lower()):
                enhanced_query = (
                    f"{user_query} [Context: Previous total budget was ${previous_budget:,.0f}]"
                )

            # Generate constraints from query (with enhanced context)
            constraints = self.generate_constraints(enhanced_query)

            # If user said "keep same budget" but constraint has different budget, override it
            if previous_budget and ("same" in user_query.lower() or "keep" in user_query.lower()):
                if (
                    abs(constraints.total_budget - previous_budget) > previous_budget * 0.1
                ):  # More than 10% difference
                    print(
                        f"\n⚠️  Correcting budget: DSPy suggested ${constraints.total_budget:,.0f}, but user requested to keep ${previous_budget:,.0f}"
                    )
                    constraints.total_budget = previous_budget

            # Run optimization
            result = self.optimize(constraints, explain=True)

            return {
                "intent": "optimization",
                "query": user_query,
                "constraints": constraints.model_dump(),
                "result": result.model_dump(),
            }
        except Exception as e:
            return {"error": f"Optimization failed: {str(e)}", "query": user_query}

    def _handle_analysis_query(self, user_query: str) -> Dict[str, Any]:
        """Handle analysis queries using model insights."""
        # Get channel performance summary
        performance = self.model.get_channel_performance_summary(self.data)

        # Format as response
        response = f"Channel Performance Summary:\n\n"
        for idx, row in performance.iterrows():
            # Check if 'channel' column exists, otherwise use index
            channel_name = row.get("channel", idx) if "channel" in performance.columns else idx

            if channel_name != "base":
                response += f"{channel_name.capitalize()}:\n"
                response += f"  - Total Spend: ${row['total_spend']:,.0f}\n"
                response += f"  - ROAS: {row['roas']:.2f}\n"
                response += f"  - % of Total Sales: {row['pct_of_total_sales']:.1f}%\n\n"

        return {
            "intent": "analysis",
            "query": user_query,
            "response": response,
            "data": performance.to_dict("index"),
        }

    def _is_genie_enabled(self) -> bool:
        """Check if Historical Genie is enabled in config."""
        genie_config = self.agent_config.get("historical_genie", {})
        return (
            genie_config.get("enabled", False)
            and genie_config.get("genie_space_id")
            and genie_config.get("genie_space_id") != "YOUR_GENIE_SPACE_ID"
        )

    def _handle_historical_genie_query(self, user_query: str) -> Dict[str, Any]:
        """Handle historical data queries via Historical Genie (fastmcp)."""
        import asyncio

        # Run async initialization and query
        return asyncio.run(self._handle_historical_genie_query_async(user_query))

    async def _handle_historical_genie_query_async(self, user_query: str) -> Dict[str, Any]:
        """Async handler for Historical Genie queries."""
        if not self._is_genie_enabled():
            return {
                "error": "Historical Genie not enabled or configured",
                "query": user_query,
                "suggestion": "Set historical_genie.enabled=true and genie_space_id in agent_config",
            }

        try:
            # Initialize Genie client if needed
            await self._initialize_historical_genie()

            if not self.historical_genie or not self.genie_client:
                return {
                    "error": "Historical Genie initialization failed",
                    "query": user_query,
                }

            print(f"\n📊 Querying Historical Genie...")

            # Use the genie client to query
            async with self.genie_client:
                # Get the query_space tool
                tools = await self.genie_client.list_tools()
                genie_tool = None
                for tool in tools:
                    if "query" in tool.name.lower():
                        genie_tool = tool
                        break

                if not genie_tool:
                    return {
                        "error": "No query tool found in Genie space",
                        "query": user_query,
                        "available_tools": [t.name for t in tools],
                    }

                # Call the Genie tool directly
                result = await self.genie_client.call_tool(
                    name=genie_tool.name, arguments={"query": user_query}
                )

                print(f"✓ Historical Genie query completed")

                return {
                    "intent": "historical_genie",
                    "query": user_query,
                    "response": str(result),
                    "data": result,
                }

        except Exception as e:
            return {
                "error": f"Historical Genie query failed: {str(e)}",
                "query": user_query,
            }

    # =========================================================================
    # Core Capability 4: Historical Data via MCP/Genie
    # =========================================================================

    def _handle_historical_data_query(self, user_query: str) -> Dict[str, Any]:
        """
        Route historical data queries to Genie space via MCP or Historical Genie.

        Note: The databricks_mcp client has async issues in Jupyter notebooks.
        For notebook use, prefer the Historical Genie integration (fastmcp).
        """
        # Check if Historical Genie is available (better for notebooks)
        if self._is_genie_enabled():
            return {
                "info": "Historical Genie is enabled. Use that for data queries in notebooks.",
                "suggestion": "Call agent._handle_historical_genie_query() or configure MCP for non-notebook use",
                "query": user_query,
            }

        if not self.mcp_client:
            return {
                "error": "MCP client not configured. Cannot access Genie spaces.",
                "query": user_query,
                "suggestion": "Configure mcp in agent_config or enable historical_genie for notebook use.",
            }

        try:
            # Get available tables from config
            catalog = self.agent_config.get("catalog", "main")
            schema = self.agent_config.get("schema", "mmm")

            available_tables = f"""
            Available tables in {catalog}.{schema}:
            - ads_sales: Historical ad spend and sales data
            - channel_contributions: Model-attributed sales by channel
            - performance_summary: ROAS and performance metrics
            - roas_comparison: Historical ROAS evolution
            """

            # Generate Genie query
            result = self.historical_data_handler(
                user_query=user_query, available_tables=available_tables
            )

            # Try to execute via MCP if available
            genie_response = self._call_genie_mcp(result.genie_query)

            return {
                "intent": "historical_data",
                "query": user_query,
                "genie_query": result.genie_query,
                "expected_insights": result.expected_insights,
                "data": genie_response,
            }

        except Exception as e:
            return {
                "error": f"Historical data query failed: {str(e)}",
                "query": user_query,
            }

    def _call_genie_mcp(self, genie_query: str) -> Any:
        """Call Genie space via MCP (synchronous wrapper for async operations)."""
        if not self.mcp_client:
            return {"error": "MCP client not available"}

        try:
            # Import nest_asyncio to handle nested event loops in Jupyter
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                pass  # nest_asyncio not available, will fail if in notebook

            # List available MCP tools (handles async internally)
            tools = self.mcp_client.list_tools()

            # Look for Genie-related tools
            # Tools may be objects or dicts, handle both
            genie_tools = []
            tool_names = []
            for t in tools:
                # Handle both dict and object formats
                if isinstance(t, dict):
                    name = t.get("name", "")
                    tool_names.append(name)
                    if "genie" in name.lower() or "query" in name.lower():
                        genie_tools.append(t)
                else:
                    # It's an object, use attribute access
                    name = getattr(t, "name", "")
                    tool_names.append(name)
                    if "genie" in name.lower() or "query" in name.lower():
                        genie_tools.append(t)

            if genie_tools:
                # Call the first Genie tool with the query
                tool = genie_tools[0]
                tool_name = tool["name"] if isinstance(tool, dict) else tool.name
                result = self.mcp_client.call_tool(tool_name, {"query": genie_query})

                # Parse the result from Genie
                # Genie returns TextContent with JSON containing conversationId, messageId, and query results
                if hasattr(result, "content"):
                    content = result.content
                    # If it's a list of TextContent objects, extract the text
                    if isinstance(content, list) and len(content) > 0:
                        import json

                        text_content = str(content[0])
                        # Try to parse as JSON to get conversation details
                        try:
                            # Extract text from TextContent
                            if "text='" in text_content:
                                json_str = text_content.split("text='")[1].split("'")[0]
                                # Unescape the JSON string
                                json_str = json_str.replace('\\"', '"').replace("\\\\", "\\")
                                genie_response = json.loads(json_str)

                                # Genie response includes conversationId, messageId, and query results
                                conversation_id = genie_response.get("conversationId")
                                message_id = genie_response.get("messageId")
                                response_content = genie_response.get("content", "")

                                if response_content:
                                    # Parse the nested JSON content with query and results
                                    try:
                                        query_data = json.loads(response_content)
                                        sql_query = query_data.get("query", "")
                                        statement_response = query_data.get(
                                            "statement_response", {}
                                        )

                                        # Extract actual data from the result
                                        result_data = statement_response.get("result", {})
                                        data_array = result_data.get("data_array", [])

                                        # Format the response
                                        formatted_result = (
                                            f"**SQL Query:**\n```sql\n{sql_query}\n```\n\n"
                                        )

                                        if data_array and len(data_array) > 0:
                                            # Get schema to understand column names
                                            manifest = statement_response.get("manifest", {})
                                            schema = manifest.get("schema", {})
                                            columns = schema.get("columns", [])

                                            formatted_result += "**Results:**\n"
                                            for row in data_array:
                                                values = row.get("values", [])
                                                for i, val_obj in enumerate(values):
                                                    col_name = (
                                                        columns[i].get("name", f"col_{i}")
                                                        if i < len(columns)
                                                        else f"col_{i}"
                                                    )
                                                    value = val_obj.get(
                                                        "string_value", val_obj.get("value", "")
                                                    )
                                                    # Format large numbers
                                                    try:
                                                        num_val = float(value)
                                                        formatted_result += (
                                                            f"- {col_name}: {num_val:,.2f}\n"
                                                        )
                                                    except (ValueError, TypeError):
                                                        formatted_result += (
                                                            f"- {col_name}: {value}\n"
                                                        )
                                        else:
                                            formatted_result += "*No data returned*\n"

                                        return {
                                            "result": formatted_result,
                                            "query": genie_query,
                                            "sql_query": sql_query,
                                            "conversation_id": conversation_id,
                                            "message_id": message_id,
                                        }
                                    except json.JSONDecodeError:
                                        # Content is not JSON, return as-is
                                        return {
                                            "result": response_content,
                                            "query": genie_query,
                                            "conversation_id": conversation_id,
                                            "message_id": message_id,
                                        }
                                else:
                                    # Content is empty, but we have conversation metadata
                                    return {
                                        "info": "Genie query initiated successfully",
                                        "conversation_id": conversation_id,
                                        "message_id": message_id,
                                        "note": "The Genie conversation was created. You may need to check the Genie space for the full response.",
                                    }
                        except (json.JSONDecodeError, IndexError) as e:
                            # Failed to parse, return raw content
                            return {
                                "result": str(content),
                                "query": genie_query,
                                "parse_error": str(e),
                            }

                    return {"result": str(content), "query": genie_query}
                else:
                    return {"result": str(result), "query": genie_query}
            else:
                return {
                    "info": "No Genie or query tools found in MCP server",
                    "available_tools": tool_names,
                }

        except RuntimeError as e:
            if "event loop" in str(e).lower():
                return {
                    "error": "MCP async error in notebook environment",
                    "suggestion": "Use the Historical Genie integration (fastmcp) instead, or run outside notebook",
                    "query": genie_query,
                }
            return {"error": f"MCP call failed: {str(e)}"}
        except Exception as e:
            return {"error": f"MCP call failed: {str(e)}"}

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_channel_insights(self) -> Dict[str, str]:
        """
        Get formatted insights for each channel.

        Returns:
            Dict mapping channel names to formatted insight strings

        Example:
            >>> insights = agent.get_channel_insights()
            >>> for channel, insight in insights.items():
            ...     print(insight)
        """
        # Get performance summary from model
        performance = self.model.get_channel_performance_summary(self.data)

        insights = {}
        for idx, row in performance.iterrows():
            channel = row["channel"]
            if channel != "base":
                insight = f"""
{channel.upper()} Performance:
• Total Spend: ${row['total_spend']:,.0f}
• ROAS: {row['roas']:.2f}x
• Contribution: {row['pct_of_total_sales']:.1f}% of total sales
• Incremental Share: {row['pct_of_incremental_sales']:.1f}% of incremental sales
                """.strip()
                insights[channel] = insight

        return insights

    def list_mcp_tools(self) -> List[Dict[str, Any]]:
        """List all available MCP tools."""
        if not self.mcp_client:
            return []

        try:
            return self.mcp_client.list_tools()
        except Exception as e:
            print(f"Error listing MCP tools: {e}")
            return []

    def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific MCP tool by name."""
        if not self.mcp_client:
            raise ValueError("MCP client not configured")

        return self.mcp_client.call_tool(tool_name, arguments)
