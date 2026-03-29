"""
Financial data nodes.

Two separate nodes serve different stages of the pipeline:

1. business_context_node — Fetches ONLY qualitative context (name, sector,
   industry, description). Used before hypothesis generation so the LLM
   constructs narratives from business understanding, not financial ratios.
   This follows Damodaran's discipline: narrative first, numbers second.

2. financial_data_node — Fetches the full quantitative snapshot (price,
   margins, multiples, implied expectations). Used in data_gathering,
   AFTER hypotheses are formed, so numbers test narratives rather than
   anchor them.
"""
from variant.state import AgentState
from variant.tools.yfinance_tools import fetch_business_context, fetch_financial_snapshot, fetch_market_context


def business_context_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    context = fetch_business_context(ticker)
    return {"business_context": context}


def financial_data_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    snapshot = fetch_financial_snapshot(ticker)
    return {"financial_data": snapshot}


def market_context_node(state: AgentState) -> dict:
    """
    Fetch multi-timeframe relative performance data for the ticker vs
    SPY, QQQ, and the sector ETF. Uses sector from business_context if available.
    """
    ticker = state["ticker"]
    bc = state.get("business_context") or {}
    sector = bc.get("sector")
    context = fetch_market_context(ticker, sector=sector)
    return {"market_context": context}
