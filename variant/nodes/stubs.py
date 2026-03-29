"""
Stub data nodes for POC.

Each stub returns structurally valid (but minimal) data so the analyst node
always receives all expected state keys. Where possible, stubs derive rough
values from the real yfinance data rather than returning empty dicts.
"""
from variant.state import AgentState


def expectations_stub_node(state: AgentState) -> dict:
    """
    Derives minimal expectations data from what yfinance already provided.
    Avoids duplicating fields already in financial_data — only adds
    derived signals (P/E compression) not present in the raw snapshot.
    """
    fd = state.get("financial_data") or {}
    forward_pe = fd.get("forward_pe")
    trailing_pe = fd.get("trailing_pe")

    # Derived signal: P/E compression indicates market expects earnings growth
    pe_compression = None
    if forward_pe and trailing_pe:
        pe_compression = round(trailing_pe - forward_pe, 1)

    return {
        "expectations_data": {
            "source": "stub_derived_from_yfinance",
            "forward_pe": forward_pe,
            "trailing_pe": trailing_pe,
            "pe_compression": pe_compression,
            "analyst_consensus": fd.get("analyst_consensus"),
            "analyst_target_price": fd.get("analyst_target_price"),
            "analyst_upside_pct": fd.get("analyst_upside_pct"),
            "num_analysts": fd.get("num_analysts"),
            "implied_expectations": fd.get("implied_expectations"),
        }
    }


def filings_stub_node(state: AgentState) -> dict:
    return {
        "filings_data": {
            "source": "stub",
        }
    }


def base_rate_stub_node(state: AgentState) -> dict:
    """Derives rough base rate flags from first principles using yfinance data."""
    fd = state.get("financial_data") or {}
    revenue_growth_pct = fd.get("revenue_growth_yoy_pct")
    market_cap_bn = fd.get("market_cap_bn")

    flags = []
    if revenue_growth_pct and revenue_growth_pct > 40:
        flags.append(
            f"Revenue growing at {revenue_growth_pct}% YoY. "
            "Historically, fewer than 10% of large-cap companies sustain >40% growth for 3+ years."
        )
    if market_cap_bn and market_cap_bn > 500 and revenue_growth_pct and revenue_growth_pct > 25:
        flags.append(
            f"Market cap ${market_cap_bn}B growing at {revenue_growth_pct}%. "
            "The law of large numbers makes high-growth persistence increasingly rare at this scale."
        )

    return {
        "base_rate_data": {
            "source": "stub_first_principles",
            "preliminary_flags": flags,
        }
    }
