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

        # Initialize DSPy modules
        if agent_config:
            self._configure_dspy(agent_config)
            self.constraint_generator = ConstraintGenerator()
            self.optimization_explainer = OptimizationExplainer()
            self.query_router = QueryRouter()
            self.historical_data_handler = HistoricalDataHandler()
        else:
            self.constraint_generator = None
            self.optimization_explainer = None
            self.query_router = None
            self.historical_data_handler = None

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

        # Silent initialization - no print statements for better UX

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
                # Silent warning - fall back to system AI tools
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

        # Silent initialization complete

    # =========================================================================
    # Core Capability 1: Generate Constraints
    # =========================================================================

    def generate_constraints(self, user_request: str, stream_callback=None) -> BudgetConstraints:
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
        return self._generate_constraints_with_dspy(user_request, stream_callback)

    def _generate_constraints_with_dspy(
        self, user_request: str, stream_callback=None
    ) -> BudgetConstraints:
        """Use DSPy to extract constraints from natural language."""

        def _stream(message: str):
            """Helper to stream messages if callback provided."""
            if stream_callback:
                stream_callback(message)

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

        _stream("💭 Analyzing budget request with LLM...")

        # Call DSPy to generate constraints with streaming
        # Note: DSPy's ChainOfThought currently doesn't support streaming to a callback
        # but we can stream status updates
        result = self.constraint_generator(
            user_request=user_request,
            historical_spend=json.dumps(historical_spend, indent=2),
            channel_performance=json.dumps(channel_performance, indent=2),
        )

        _stream("✓ Constraints extracted from LLM response")

        # Parse the response
        try:
            total_budget = float(result.total_budget)
            min_spend_dict = json.loads(result.min_spend_per_channel)
            max_spend_dict = json.loads(result.max_spend_per_channel)

            # Build nested channels dict for BudgetConstraints
            channels = {}
            for ch in channel_names:
                channels[ch] = {
                    "min_spend": min_spend_dict.get(ch, 0),
                    "max_spend": max_spend_dict.get(ch, total_budget),
                }

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
        self,
        constraints: Optional[BudgetConstraints] = None,
        explain: bool = True,
        stream_callback=None,
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
                constraints, opt_result, channel_roas, stream_callback
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
        stream_callback=None,
    ) -> tuple[str, str]:
        """Generate natural language explanation of optimization."""

        def _stream(message: str):
            """Helper to stream messages if callback provided."""
            if stream_callback:
                stream_callback(message)

        # Get current allocation from historical data
        channel_names = [c.name for c in self.model.config.channels]
        current_allocation = {
            ch: round(float(self.data[ch].mean()), 1)
            for ch in channel_names
            if ch in self.data.columns
        }

        # Round ROAS values to 1 decimal place
        rounded_channel_roas = {ch: round(roas, 1) for ch, roas in channel_roas.items()}

        # Format for DSPy with rounded values
        performance_metrics = {
            "channel_roas": rounded_channel_roas,
            "total_roas": round(
                sum(opt_result.channel_contributions.values())
                / sum(opt_result.optimal_allocation.values()),
                1,
            ),
            "expected_sales": round(sum(opt_result.channel_contributions.values()), 1),
        }

        # Round optimal allocation values
        rounded_optimal_allocation = {
            ch: round(spend, 1) for ch, spend in opt_result.optimal_allocation.items()
        }

        _stream("📝 Generating explanation with LLM...")

        result = self.optimization_explainer(
            user_request=f"Optimize budget with total of ${constraints.total_budget:,.0f}",
            current_allocation=json.dumps(current_allocation, indent=2),
            optimal_allocation=json.dumps(rounded_optimal_allocation, indent=2),
            performance_metrics=json.dumps(performance_metrics, indent=2),
        )

        _stream("✓ Explanation generated")

        return result.explanation, result.recommendations

    # =========================================================================
    # Core Capability 3: End-to-End Query Handler
    # =========================================================================

    def query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream_callback=None,
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
            stream_callback: Optional callback function to stream reasoning steps (takes a string message)

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

        def _stream(message: str):
            """Helper to stream messages if callback provided."""
            if stream_callback:
                stream_callback(message)

        if not self.query_router:
            return {
                "error": "DSPy not configured. Please provide agent_config.",
                "query": user_query,
            }

        _stream("🤔 Understanding your query...")

        # Build context from conversation history
        context_str = ""
        if conversation_history:
            _stream("📚 Reviewing conversation history...")
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
        _stream("🧭 Determining query intent with LLM...")
        capabilities = """
        - optimization: Budget allocation optimization with constraints
        - historical_data: Query historical sales and spend data via Genie/Unity Catalog
        - historical_genie: Query historical data using Genie space (for peak sales, trends, etc.)
        - analysis: Analyze channel performance, ROAS, contributions
        """

        routing = self.query_router(
            user_query=query_with_context, available_capabilities=capabilities
        )

        _stream(f"✓ Intent identified: {routing.intent}")

        # Handle based on intent
        if routing.intent == "optimization":
            return self._handle_optimization_query(
                user_query, conversation_history, stream_callback
            )
        elif routing.intent == "historical_data" and routing.requires_mcp:
            return self._handle_historical_data_query(user_query, stream_callback)
        elif routing.intent == "analysis":
            return self._handle_analysis_query(user_query, stream_callback)
        else:
            return {"response": "I'm not sure how to handle that query.", "intent": routing.intent}

    def _handle_optimization_query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream_callback=None,
    ) -> Dict[str, Any]:
        """Handle optimization-related queries with conversation context."""

        def _stream(message: str):
            """Helper to stream messages if callback provided."""
            if stream_callback:
                stream_callback(message)

        try:
            # Look for previous optimization results in conversation history
            previous_budget = None
            previous_allocation = None

            if conversation_history:
                _stream("🔍 Extracting context from previous conversation...")
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
                                    _stream(f"✓ Found previous budget: ${previous_budget:,.0f}")
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
            _stream("🎯 Generating optimization constraints...")
            constraints = self.generate_constraints(enhanced_query, stream_callback)

            # If user said "keep same budget" but constraint has different budget, override it
            if previous_budget and ("same" in user_query.lower() or "keep" in user_query.lower()):
                if (
                    abs(constraints.total_budget - previous_budget) > previous_budget * 0.1
                ):  # More than 10% difference
                    _stream(
                        f"⚠️ Overriding DSPy budget ${constraints.total_budget:,.0f} with previous budget ${previous_budget:,.0f}"
                    )
                    constraints.total_budget = previous_budget

            # Run optimization
            _stream("⚙️ Running optimization algorithm...")
            result = self.optimize(constraints, explain=True, stream_callback=stream_callback)

            _stream("✓ Optimization complete!")

            return {
                "intent": "optimization",
                "query": user_query,
                "constraints": constraints.model_dump(),
                "result": result.model_dump(),
            }
        except Exception as e:
            return {"error": f"Optimization failed: {str(e)}", "query": user_query}

    def _handle_analysis_query(self, user_query: str, stream_callback=None) -> Dict[str, Any]:
        """Handle analysis queries using model insights."""

        def _stream(message: str):
            """Helper to stream messages if callback provided."""
            if stream_callback:
                stream_callback(message)

        _stream("📊 Analyzing channel performance...")
        # Get channel performance summary
        performance = self.model.get_channel_performance_summary(self.data)

        _stream("✓ Performance metrics calculated")

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

    # =========================================================================
    # Core Capability 4: Historical Data via MCP/Genie
    # =========================================================================

    def _handle_historical_data_query(
        self, user_query: str, stream_callback=None
    ) -> Dict[str, Any]:
        """Route historical data queries to Genie space via MCP."""

        def _stream(message: str):
            """Helper to stream messages if callback provided."""
            if stream_callback:
                stream_callback(message)

        if not self.mcp_client:
            return {
                "error": "MCP client not configured. Cannot access Genie spaces.",
                "query": user_query,
                "suggestion": "Configure mcp in agent_config with genie_space_id.",
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

            _stream("🔍 Translating query to SQL...")
            # Generate Genie query
            result = self.historical_data_handler(
                user_query=user_query, available_tables=available_tables
            )

            _stream(f"✓ Generated SQL query")
            _stream("📊 Executing query via Genie...")
            # Try to execute via MCP if available
            genie_response = self._call_genie_mcp(result.genie_query, stream_callback)

            _stream("✓ Query complete!")

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

    def _call_genie_mcp(self, genie_query: str, stream_callback=None) -> Any:
        """Call Genie space via direct HTTP API (fully synchronous)."""

        def _stream(message: str):
            """Helper to stream messages if callback provided."""
            if stream_callback:
                stream_callback(message)

        if not self.workspace_client:
            return {"error": "Workspace client not available"}

        try:
            # Get Genie space ID from config
            mcp_config = self.agent_config.get("mcp", {})
            genie_space_id = mcp_config.get("genie_space_id")

            if not genie_space_id or genie_space_id == "YOUR_GENIE_SPACE_ID":
                return {
                    "error": "Genie space ID not configured",
                    "suggestion": "Set mcp.genie_space_id in your config",
                }

            # Use requests for synchronous HTTP calls
            import requests
            import json

            # Get workspace host and auth token
            host = self.workspace_client.config.host

            # Get authentication token - try multiple methods
            token = None
            if (
                hasattr(self.workspace_client.config, "token")
                and self.workspace_client.config.token
            ):
                token = self.workspace_client.config.token
            elif hasattr(self.workspace_client.config, "oauth_token"):
                # For OAuth, get the access token
                oauth = self.workspace_client.config.oauth_token()
                token = oauth.access_token if hasattr(oauth, "access_token") else str(oauth)

            if not token:
                return {"error": "Could not retrieve authentication token"}

            # Construct the Genie API endpoint
            url = f"{host}/api/2.0/genie/spaces/{genie_space_id}/start-conversation"

            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

            payload = {"content": genie_query}

            # Make synchronous HTTP request
            _stream("🚀 Sending query to Genie...")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result_data = response.json()

            # Extract conversation details
            conversation_id = result_data.get("conversation_id")
            message_id = result_data.get("message_id") or result_data.get("message", {}).get("id")

            # Genie processes queries asynchronously, need to poll for results
            import time

            max_retries = 30  # 30 seconds max
            retry_count = 0

            message_url = f"{host}/api/2.0/genie/spaces/{genie_space_id}/conversations/{conversation_id}/messages/{message_id}"

            _stream("⏳ Waiting for Genie to process query...")
            while retry_count < max_retries:
                time.sleep(1)  # Wait 1 second between polls
                retry_count += 1

                message_response = requests.get(message_url, headers=headers, timeout=30)
                message_response.raise_for_status()
                message_data = message_response.json()

                status = message_data.get("status")

                # Stream progress updates
                if retry_count % 5 == 0:  # Every 5 seconds
                    _stream(f"⏳ Still processing... ({retry_count}s)")

                if status == "COMPLETED":
                    break
                elif status in ["FAILED", "CANCELLED"]:
                    return {
                        "error": f"Genie query failed with status: {status}",
                        "query": genie_query,
                    }

            if retry_count >= max_retries:
                return {
                    "error": "Genie query timed out waiting for results",
                    "query": genie_query,
                    "conversation_id": conversation_id,
                }

            # Extract results from completed message
            result_data = message_data

            # The response might include attachments with SQL and results
            attachments = result_data.get("attachments", [])

            formatted_result = ""
            sql_query = None

            # Look for SQL query and results in attachments
            for attachment in attachments:
                if attachment.get("query"):
                    query_info = attachment["query"]
                    sql_query = query_info.get("query")
                    statement_id = query_info.get("statement_id")

                    if sql_query:
                        formatted_result += f"**SQL Query:**\n```sql\n{sql_query}\n```\n\n"

                    # Get query results - check both "result" and "data"
                    result_info = query_info.get("result") or query_info.get("data", {})

                    # If we have a statement_id, fetch the actual results
                    if statement_id and not result_info:
                        try:
                            # Use SQL execution API to get results
                            results_url = (
                                f"{host}/api/2.0/sql/statements/{statement_id}/result/chunks/0"
                            )
                            results_response = requests.get(
                                results_url, headers=headers, timeout=30
                            )
                            results_response.raise_for_status()
                            results_json = results_response.json()

                            # Extract data from the results
                            if "data_array" in results_json:
                                data_array = results_json["data_array"]
                                manifest = results_json.get("manifest", {})

                                if data_array:
                                    # Try to get schema from manifest or query_info
                                    schema = manifest.get("schema", {})
                                    columns = schema.get("columns", [])

                                    # If no columns in manifest, try from query_result_metadata
                                    if not columns and "query_result_metadata" in query_info:
                                        metadata = query_info["query_result_metadata"]
                                        if "columns" in metadata:
                                            columns = metadata["columns"]

                                    # Format as markdown table
                                    formatted_result += "**Results:**\n\n"

                                    # Determine column names from schema
                                    if columns and isinstance(columns, list) and len(columns) > 0:
                                        # Extract column names from schema
                                        col_names = []
                                        for col in columns:
                                            if isinstance(col, dict):
                                                col_names.append(col.get("name", "unknown"))
                                            else:
                                                col_names.append(str(col))
                                    else:
                                        # Fallback: try to extract from SQL query
                                        col_names = []
                                        if sql_query:
                                            # Parse SELECT columns from SQL
                                            import re

                                            # Look for SELECT ... FROM pattern
                                            select_match = re.search(
                                                r"SELECT\s+(.*?)\s+FROM",
                                                sql_query,
                                                re.IGNORECASE | re.DOTALL,
                                            )
                                            if select_match:
                                                select_clause = select_match.group(1)
                                                # Extract column names (handle backticks, aliases, etc.)
                                                col_parts = [
                                                    c.strip() for c in select_clause.split(",")
                                                ]
                                                for part in col_parts:
                                                    # Remove backticks and extract just the column name
                                                    col_name = part.replace("`", "").strip()
                                                    # Handle aliases (AS keyword or space-separated)
                                                    if " as " in col_name.lower():
                                                        col_name = (
                                                            col_name.lower()
                                                            .split(" as ")[-1]
                                                            .strip()
                                                        )
                                                    elif " " in col_name and not any(
                                                        func in col_name.upper()
                                                        for func in [
                                                            "COUNT",
                                                            "SUM",
                                                            "AVG",
                                                            "MAX",
                                                            "MIN",
                                                        ]
                                                    ):
                                                        # Space-separated alias
                                                        col_name = col_name.split()[-1]
                                                    # Extract just the column name (after table prefix if any)
                                                    if "." in col_name:
                                                        col_name = col_name.split(".")[-1]
                                                    col_names.append(col_name)

                                        # If still no columns, use generic names
                                        if not col_names:
                                            first_row = data_array[0] if data_array else []
                                            if isinstance(first_row, list):
                                                num_cols = len(first_row)
                                            else:
                                                num_cols = len(first_row.get("values", []))
                                            col_names = [f"col_{i}" for i in range(num_cols)]

                                    formatted_result += "| " + " | ".join(col_names) + " |\n"
                                    formatted_result += (
                                        "|" + "|".join(["---" for _ in col_names]) + "|\n"
                                    )

                                    # Add data rows
                                    for row in data_array:
                                        # Each row is a list of values
                                        if isinstance(row, list):
                                            formatted_values = []
                                            for idx, value in enumerate(row):
                                                # Handle NULL values
                                                if value is None:
                                                    formatted_values.append("(null)")
                                                else:
                                                    # Format large numbers
                                                    try:
                                                        num_val = float(value)
                                                        formatted_values.append(f"{num_val:,.2f}")
                                                    except (ValueError, TypeError):
                                                        # Could be a date/string
                                                        formatted_values.append(str(value))
                                            formatted_result += (
                                                "| " + " | ".join(formatted_values) + " |\n"
                                            )
                                        else:
                                            # Dict format (old path) - also format as table
                                            values = row.get("values", [])
                                            formatted_values = []
                                            for idx, val_obj in enumerate(values):
                                                value = val_obj.get("str_value") or val_obj.get(
                                                    "value", ""
                                                )
                                                # Format large numbers
                                                try:
                                                    num_val = float(value)
                                                    formatted_values.append(f"{num_val:,.2f}")
                                                except (ValueError, TypeError):
                                                    formatted_values.append(str(value))
                                            formatted_result += (
                                                "| " + " | ".join(formatted_values) + " |\n"
                                            )
                                else:
                                    formatted_result += "*No data returned*\n"
                        except Exception as e:
                            formatted_result += f"*Error fetching results: {str(e)}*\n"
                    else:
                        # Try to get data from the query_info directly
                        data_array = result_info.get("data_array", [])

                        if data_array:
                            # Get schema for column names
                            manifest = query_info.get("manifest", {})
                            schema = manifest.get("schema", {})
                            columns = schema.get("columns", [])

                            # Also try query_result_metadata
                            if not columns and "query_result_metadata" in query_info:
                                metadata = query_info["query_result_metadata"]
                                columns = metadata.get("columns", [])

                            # Format as markdown table
                            formatted_result += "**Results:**\n\n"

                            # Extract column names from schema
                            if columns and isinstance(columns, list) and len(columns) > 0:
                                col_names = []
                                for col in columns:
                                    if isinstance(col, dict):
                                        col_names.append(col.get("name", "unknown"))
                                    else:
                                        col_names.append(str(col))
                            else:
                                # Fallback: try to extract from SQL query
                                col_names = []
                                if sql_query:
                                    # Parse SELECT columns from SQL
                                    import re

                                    # Look for SELECT ... FROM pattern
                                    select_match = re.search(
                                        r"SELECT\s+(.*?)\s+FROM",
                                        sql_query,
                                        re.IGNORECASE | re.DOTALL,
                                    )
                                    if select_match:
                                        select_clause = select_match.group(1)
                                        # Extract column names (handle backticks, aliases, etc.)
                                        col_parts = [c.strip() for c in select_clause.split(",")]
                                        for part in col_parts:
                                            # Remove backticks and extract just the column name
                                            col_name = part.replace("`", "").strip()
                                            # Handle aliases (AS keyword or space-separated)
                                            if " as " in col_name.lower():
                                                col_name = (
                                                    col_name.lower().split(" as ")[-1].strip()
                                                )
                                            elif " " in col_name and not any(
                                                func in col_name.upper()
                                                for func in ["COUNT", "SUM", "AVG", "MAX", "MIN"]
                                            ):
                                                # Space-separated alias
                                                col_name = col_name.split()[-1]
                                            # Extract just the column name (after table prefix if any)
                                            if "." in col_name:
                                                col_name = col_name.split(".")[-1]
                                            col_names.append(col_name)

                                # If still no columns, use generic names
                                if not col_names:
                                    first_row = data_array[0] if data_array else {}
                                    num_cols = len(first_row.get("values", []))
                                    col_names = [f"col_{i}" for i in range(num_cols)]

                            formatted_result += "| " + " | ".join(col_names) + " |\n"
                            formatted_result += "|" + "|".join(["---" for _ in col_names]) + "|\n"

                            # Add data rows
                            for row in data_array:
                                values = row.get("values", [])
                                formatted_values = []
                                for idx, val_obj in enumerate(values):
                                    value = val_obj.get("string_value", val_obj.get("value", ""))

                                    # Handle NULL values
                                    if value is None or value == "":
                                        formatted_values.append("(null)")
                                    else:
                                        # Format large numbers
                                        try:
                                            num_val = float(value)
                                            formatted_values.append(f"{num_val:,.2f}")
                                        except (ValueError, TypeError):
                                            formatted_values.append(str(value))

                                formatted_result += "| " + " | ".join(formatted_values) + " |\n"
                        else:
                            formatted_result += "*No data returned*\n"
                elif attachment.get("text"):
                    # Text attachment, might be the result message
                    text_content = attachment["text"].get("content", "")
                    if text_content and not formatted_result:
                        formatted_result = text_content

            # If we found formatted results, return them
            if formatted_result:
                return {
                    "result": formatted_result,
                    "query": genie_query,
                    "sql_query": sql_query,
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                }

            # Otherwise return the raw message content
            message = result_data.get("message", "")
            if message:
                return {
                    "result": message,
                    "query": genie_query,
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                }

            # Fallback: return raw response
            return {
                "result": json.dumps(result_data, indent=2),
                "query": genie_query,
                "conversation_id": conversation_id,
            }

        except requests.exceptions.RequestException as e:
            return {"error": f"Genie API call failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Genie call failed: {str(e)}"}

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
            # Silent error handling
            return []

    def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific MCP tool by name."""
        if not self.mcp_client:
            raise ValueError("MCP client not configured")

        return self.mcp_client.call_tool(tool_name, arguments)
