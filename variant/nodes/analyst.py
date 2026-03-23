"""
Analyst node — the core intelligence of the system.

This is where the real analytical work happens. The analyst receives:
- The hypothesis generator's narratives (bull/base/bear)
- All gathered data (financial, expectations, news, filings, base rates)
- The current iteration count

It then works through a 5-step framework:
1. Narrative Evaluation — Test each narrative against data using Damodaran's 3P test
2. Expectations Gap — Reverse-engineer what the price implies (Mauboussin)
3. Base Rate Check — Flag statistically unusual assumptions
4. Contradiction Detection — Find conflicting signals in the data
5. Loop Decision — Decide whether another data-gathering pass would help

The output is structured (via Pydantic) to ensure all downstream nodes
receive the exact fields they expect.
"""
import json
from typing import Optional

from pydantic import BaseModel, Field

from variant.state import AgentState, Narrative
from variant.prompts.analyst import SYSTEM_PROMPT, USER_TEMPLATE
from variant.config import get_llm


# ── Pydantic schemas for structured LLM output ────────────────────────

class NarrativeResult(BaseModel):
    """A narrative after analyst evaluation — same shape as Narrative but with evidence filled in."""
    label: str
    story: str
    probability: float
    key_assumptions: list[str]
    supporting_evidence: list[str]
    contradicting_evidence: list[str]


class ExpectationsGap(BaseModel):
    """What the current stock price implies vs. what we think is likely."""
    price_implied_growth_pct: Optional[float] = None
    price_implied_margin_pct: Optional[float] = None
    closest_narrative: str          # Which scenario (bull/base/bear) the market is pricing
    gap_assessment: str             # 1-2 sentence assessment of where market may be wrong


class AnalystOutput(BaseModel):
    """Complete output from the analyst's 5-step analysis."""
    updated_narratives: list[NarrativeResult]
    expectations_gap: ExpectationsGap
    contradictions: list[str]
    base_rate_flags: list[str]
    follow_up_questions: list[str]
    needs_more_data: bool                   # Loop decision
    analyst_reasoning_summary: str          # Key insight for synthesis


def _format_financial_data(fd: dict | None) -> str:
    """Format financial data as readable lines, not raw JSON."""
    if not fd:
        return "Not available."
    if fd.get("error"):
        return f"Error: {fd['error']}"
    lines = []
    if fd.get("company_name"):
        lines.append(f"Company: {fd['company_name']} ({fd.get('sector', 'N/A')} / {fd.get('industry', 'N/A')})")
    if fd.get("current_price"):
        lines.append(f"Price: ${fd['current_price']} | Market Cap: ${fd.get('market_cap_bn', 'N/A')}B | EV: ${fd.get('enterprise_value_bn', 'N/A')}B")
    if fd.get("revenue_bn_ttm"):
        lines.append(f"Revenue (TTM): ${fd['revenue_bn_ttm']}B | Growth YoY: {fd.get('revenue_growth_yoy_pct', 'N/A')}%")
    if fd.get("ebitda_bn_ttm"):
        lines.append(f"EBITDA (TTM): ${fd['ebitda_bn_ttm']}B | FCF: ${fd.get('free_cash_flow_bn_ttm', 'N/A')}B")
    if fd.get("operating_margin_pct") is not None:
        lines.append(f"Margins: Gross {fd.get('gross_margin_pct', 'N/A')}% | Operating {fd['operating_margin_pct']}% | Net {fd.get('net_margin_pct', 'N/A')}%")
    if fd.get("forward_pe"):
        lines.append(f"Valuation: Forward P/E {fd['forward_pe']}x | Trailing P/E {fd.get('trailing_pe', 'N/A')}x | EV/EBITDA {fd.get('ev_to_ebitda', 'N/A')}x | P/S {fd.get('price_to_sales', 'N/A')}x")
    if fd.get("analyst_consensus"):
        lines.append(f"Consensus: {fd['analyst_consensus']} (n={fd.get('num_analysts', '?')}) | Target: ${fd.get('analyst_target_price', 'N/A')} ({fd.get('analyst_upside_pct', 'N/A')}% upside)")
    if fd.get("pct_from_52w_high") is not None:
        lines.append(f"52w range: ${fd.get('price_52w_low', '?')} – ${fd.get('price_52w_high', '?')} (currently {fd['pct_from_52w_high']}% from high)")
    # EPS surprises
    eps = fd.get("eps_surprise_history", [])
    if eps:
        surprises = [f"{e.get('surprise_pct', '?')}%" for e in eps if e.get("surprise_pct") is not None]
        if surprises:
            lines.append(f"Recent EPS surprises: {', '.join(surprises)} (last {len(surprises)} quarters)")
    # Implied expectations (reverse DCF)
    ie = fd.get("implied_expectations")
    if ie:
        lines.append(f"Implied expectations (reverse DCF): Market prices in {ie.get('implied_revenue_cagr_pct', '?')}% revenue CAGR over {ie['assumptions']['projection_years']}yr")
        lines.append(f"  → Implies terminal revenue of ${ie.get('implied_terminal_revenue_bn', '?')}B (vs current ${fd.get('revenue_bn_ttm', '?')}B)")
        lines.append(f"  → Assumes: WACC {ie['assumptions']['wacc_pct']}%, terminal growth {ie['assumptions']['terminal_growth_pct']}%, margin held at {ie['assumptions']['operating_margin_held_at_pct']}%")
    return "\n".join(lines) if lines else "No financial data available."


def _format_expectations(ed: dict | None) -> str:
    if not ed or ed.get("source") == "stub":
        return "Not available (no real-time consensus data in POC)."
    lines = []
    if ed.get("forward_pe"):
        lines.append(f"Forward P/E: {ed['forward_pe']}x | Trailing P/E: {ed.get('trailing_pe', 'N/A')}x")
    pe_comp = ed.get("pe_compression")
    if pe_comp is not None:
        direction = "compression (market expects earnings growth)" if pe_comp > 0 else "expansion (market expects earnings decline)"
        lines.append(f"P/E {direction}: {pe_comp} point gap")
    if ed.get("analyst_consensus"):
        lines.append(f"Analyst consensus: {ed['analyst_consensus']} | Target: ${ed.get('analyst_target_price', 'N/A')} ({ed.get('analyst_upside_pct', 'N/A')}% upside, n={ed.get('num_analysts', '?')})")
    return "\n".join(lines) if lines else "Minimal expectations data available."


def _format_simple_stub(data: dict | None, label: str) -> str:
    if not data or data.get("source") == "stub":
        return f"{label}: Not available in POC."
    # For base_rate_data, surface the preliminary flags
    flags = data.get("preliminary_flags", [])
    if flags:
        return "\n".join(f"- {f}" for f in flags)
    return f"{label}: No significant findings."


def analyst_node(state: AgentState) -> dict:
    llm = get_llm(structured_output_schema=AnalystOutput)

    user_message = USER_TEMPLATE.format(
        ticker=state["ticker"],
        query=state["query"],
        narratives_json=json.dumps(state.get("narratives", []), indent=2),
        financial_data_json=_format_financial_data(state.get("financial_data")),
        expectations_data_json=_format_expectations(state.get("expectations_data")),
        news_sentiment_json=_format_simple_stub(state.get("news_sentiment"), "News & Sentiment"),
        filings_data_json=_format_simple_stub(state.get("filings_data"), "SEC Filings"),
        base_rate_data_json=_format_simple_stub(state.get("base_rate_data"), "Base Rates"),
        iteration=state.get("iteration", 0),
    )

    result: AnalystOutput = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    updated_narratives: list[Narrative] = [
        {
            "label": n.label,
            "story": n.story,
            "probability": n.probability,
            "key_assumptions": n.key_assumptions,
            "supporting_evidence": n.supporting_evidence,
            "contradicting_evidence": n.contradicting_evidence,
        }
        for n in result.updated_narratives
    ]

    return {
        "narratives": updated_narratives,
        "expectations_gap": result.expectations_gap.model_dump(),
        "contradictions": result.contradictions,
        "base_rate_flags": result.base_rate_flags,
        "follow_up_questions": result.follow_up_questions,
        "needs_more_data": result.needs_more_data,
        "iteration": state.get("iteration", 0) + 1,
        "analyst_reasoning_summary": result.analyst_reasoning_summary,
    }
