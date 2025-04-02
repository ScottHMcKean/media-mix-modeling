from langgraph.graph import Graph, StateGraph, START, END
from langgraph.graph import StateGraph
import mlflow
from openai import OpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    FunctionMessage,
    AIMessage,
)
from pydantic import BaseModel
from mediamix.nodes import (
    AgentState,
    QueryCategorizer,
    route_query,
    clarify_intent,
    optimize_budget,
    analyze_channels,
    analyze_constraints,
    generate_forecast,
    fetch_historical,
)


def create_initial_state(query: str) -> AgentState:
    """Create initial state for the agent"""
    return {
        "messages": [HumanMessage(content=query)],
        "query_category": None,
        "query_category_confidence": None,
        "query_category_reasoning": None,
        "current_step": "start",
        "error": None,
    }


def create_mmm_agent(
    client: OpenAI,
    config: mlflow.models.model_config.ModelConfig,
) -> Graph:
    """
    Create the marketing mix modeling agent workflow
    """
    # state
    workflow = StateGraph(AgentState)

    # nodes
    workflow.add_node(
        "categorize_query",
        QueryCategorizer(client, config.get("query_categorizer")),
    )

    workflow.add_node("clarify_intent", clarify_intent)
    workflow.add_node("optimize_budget", optimize_budget)
    workflow.add_node("analyze_channels", analyze_channels)
    workflow.add_node("analyze_constraints", analyze_constraints)
    workflow.add_node("generate_forecast", generate_forecast)
    workflow.add_node("fetch_historical", fetch_historical)

    # edges
    workflow.add_edge(START, "categorize_query")
    workflow.add_conditional_edges(
        "categorize_query",
        route_query,
        {
            "budget_optimization": "optimize_budget",
            "channel_performance": "analyze_channels",
            "constraint_analysis": "analyze_constraints",
            "future_forecast": "generate_forecast",
            "historical_data": "fetch_historical",
            "clarify_intent": "clarify_intent",
        },
    )
    workflow.add_edge("clarify_intent", END)
    workflow.add_edge("optimize_budget", END)
    workflow.add_edge("analyze_channels", END)
    workflow.add_edge("analyze_constraints", END)
    workflow.add_edge("generate_forecast", END)
    workflow.add_edge("fetch_historical", END)

    # compile
    chain = workflow.compile()

    return chain
