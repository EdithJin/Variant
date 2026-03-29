"""
Shared state schema for the Variant agent graph.

All nodes read from and write to AgentState. LangGraph manages the state
transitions — each node returns a dict of keys to update, and LangGraph
merges them into the current state before passing it to the next node.
"""
from typing import TypedDict, Optional


class Narrative(TypedDict):
    """
    A single competing narrative (bull, base, or bear case).
    Created by the hypothesis_generator, then updated by the analyst
    with evidence and revised probabilities.
    """
    label: str                        # "bull", "base", or "bear"
    story: str                        # 2-3 sentence narrative about the company's future
    probability: float                # 0.0-1.0, all three must sum to ~1.0
    key_assumptions: list[str]        # What must be true for this narrative to play out
    supporting_evidence: list[str]    # Data points that support this narrative (filled by analyst)
    contradicting_evidence: list[str] # Data points that contradict it (filled by analyst)


class AgentState(TypedDict):
    """
    The shared state that flows through the entire LangGraph.
    Each node reads what it needs and returns a partial dict of updates.
    """
    # ── Input (set once at the start) ──────────────────────────────
    query: str                          # User's original question
    ticker: str                         # Stock ticker symbol (e.g., "NVDA")

    # ── Hypothesis Generation ──────────────────────────────────────
    narratives: list[Narrative]         # 3 competing narratives (bull/base/bear)

    # ── Business Context (populated before hypothesis generation) ───
    business_context: Optional[dict]    # Company name, sector, industry, description (no financials)

    # ── Gathered Data (populated by data sub-agents AFTER hypotheses) ─
    # In POC: financial_data and market_context are real (yfinance), others are stubs
    financial_data: Optional[dict]      # Price, ratios, margins, consensus
    market_context: Optional[dict]      # Multi-timeframe relative perf vs SPY/QQQ/sector ETF + VIX
    expectations_data: Optional[dict]   # Market consensus estimates
    news_sentiment: Optional[dict]      # Recent news and sentiment
    filings_data: Optional[dict]        # SEC filings analysis
    base_rate_data: Optional[dict]      # Historical base rates for growth/margins

    # ── Analyst Reasoning (populated by analyst node) ──────────────
    expectations_gap: Optional[dict]    # Price-implied growth vs. narrative scenarios
    contradictions: list[str]           # Data contradictions found
    base_rate_flags: list[str]          # Flags for statistically unusual assumptions
    follow_up_questions: list[str]      # Questions that better data might answer

    # ── Control Flow ───────────────────────────────────────────────
    iteration: int                      # Current analysis iteration (0-based, max 3)
    needs_more_data: bool               # Analyst's decision: loop back or proceed

    # ── Output ─────────────────────────────────────────────────────
    analyst_reasoning_summary: Optional[str]  # Key insight (passed to synthesis)
    final_brief: Optional[str]                # Formatted research brief
