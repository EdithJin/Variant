"""
LangGraph graph assembly for Variant.

This module defines the agent's control flow as a directed graph:

    business_context → hypothesis_generator → data_gathering → analyst
                                                    ↑              │
                                                    └── loop ──────┘
                                                                   │
                                                             synthesis → END

The graph has one conditional edge: after the analyst node, it either loops back
to data_gathering (if the analyst needs more information) or proceeds to synthesis.
The loop is capped at 3 iterations to prevent runaway execution.

There is no query classification step. The LLM reads the user's query directly
and adapts its analysis accordingly — this is more robust than keyword matching
and is what makes the system genuinely agentic.
"""
from langgraph.graph import StateGraph, END

from variant.state import AgentState
from variant.nodes.financial_data import business_context_node, financial_data_node
from variant.nodes.hypothesis_generator import hypothesis_generator_node
from variant.nodes.stubs import (
    expectations_stub_node,
    news_stub_node,
    filings_stub_node,
    base_rate_stub_node,
)
from variant.nodes.analyst import analyst_node
from variant.nodes.synthesis import synthesis_node


def data_gathering_node(state: AgentState) -> dict:
    """
    Coordinator node that runs all data sub-agents sequentially and merges results.

    In the POC, only financial_data_node fetches real data (via yfinance).
    The other nodes return stub data so the analyst always receives all expected
    state keys. Phase 2 will replace stubs with real implementations and run
    them in parallel using LangGraph's Send API.

    On loop iterations, skips re-fetching financial data since it won't change.
    """
    updates = {}

    # Only fetch financial data on first pass (won't change on re-runs)
    if not state.get("financial_data"):
        updates.update(financial_data_node(state))

    # Stubs need the latest financial data to derive their placeholder values
    stub_state = {**state, **updates}
    updates.update(expectations_stub_node(stub_state))
    updates.update(news_stub_node(stub_state))
    updates.update(filings_stub_node(stub_state))
    updates.update(base_rate_stub_node(stub_state))

    return updates


def should_loop(state: AgentState) -> str:
    """
    Conditional edge: decides whether the analyst needs another data-gathering pass.

    Returns "data_gathering" to loop back, or "synthesis" to proceed to output.
    The analyst sets needs_more_data=true only when it has a specific question
    that additional data would answer. The 3-iteration cap is a safety net.
    """
    if state.get("needs_more_data") and state.get("iteration", 0) < 3:
        return "data_gathering"
    return "synthesis"


def build_graph() -> StateGraph:
    """
    Assembles and compiles the full Variant agent graph.

    Node execution order (linear path, no loop):
        business_context → hypothesis_generator → data_gathering → analyst → synthesis

    business_context fetches ONLY qualitative info (company name, sector,
    industry, description) — no financial ratios or multiples. This follows
    Damodaran's framework: construct narratives from business understanding
    first, then gather numbers to test them. The full financial snapshot
    is fetched in data_gathering, AFTER hypotheses are formed.

    There is no query classification node. The LLM reads the user's query
    directly in the hypothesis generator and analyst prompts, and adapts
    its focus naturally. This is more robust than keyword matching and
    avoids the false precision of categorizing free-form questions.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────
    graph.add_node("business_context", business_context_node)
    graph.add_node("hypothesis_generator", hypothesis_generator_node)
    graph.add_node("data_gathering", data_gathering_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("synthesis", synthesis_node)

    # ── Edges (define execution order) ──────────────────────────────
    graph.set_entry_point("business_context")
    graph.add_edge("business_context", "hypothesis_generator")
    graph.add_edge("hypothesis_generator", "data_gathering")
    graph.add_edge("data_gathering", "analyst")

    # After analyst: either loop back for more data or proceed to synthesis
    graph.add_conditional_edges(
        "analyst",
        should_loop,
        {
            "data_gathering": "data_gathering",
            "synthesis": "synthesis",
        },
    )

    graph.add_edge("synthesis", END)

    return graph.compile()
