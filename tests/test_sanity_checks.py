"""
Tests for pipeline sanity checks.

These verify that run_sanity_checks() correctly identifies valid and invalid
agent states — catching broken pipelines without false positives.
"""
import pytest
from variant.evaluate import run_sanity_checks


def _valid_state():
    """A minimal but structurally valid agent state."""
    return {
        "query": "Is AAPL overvalued?",
        "ticker": "AAPL",
        "narratives": [
            {"label": "bull", "story": "Growth story", "probability": 0.3,
             "key_assumptions": ["A"], "supporting_evidence": ["B"], "contradicting_evidence": []},
            {"label": "base", "story": "Steady story", "probability": 0.4,
             "key_assumptions": ["C"], "supporting_evidence": ["D"], "contradicting_evidence": []},
            {"label": "bear", "story": "Decline story", "probability": 0.3,
             "key_assumptions": ["E"], "supporting_evidence": [], "contradicting_evidence": ["F"]},
        ],
        "financial_data": {
            "ticker": "AAPL",
            "current_price": 180.0,
            "market_cap_bn": 2800.0,
            "revenue_bn_ttm": 385.0,
            "implied_expectations": {
                "implied_revenue_cagr_pct": 8.5,
                "assumptions": {"projection_years": 5, "wacc_pct": 9.0,
                                "terminal_growth_pct": 3.0, "operating_margin_held_at_pct": 30.0},
            },
        },
        "expectations_gap": {
            "closest_narrative": "base",
            "gap_assessment": "Market expectations align with base case.",
            "price_implied_growth_pct": 8.5,
        },
        "final_brief": "VARIANT RESEARCH BRIEF: AAPL\n...",
    }


def test_valid_state_passes_all():
    result = run_sanity_checks(_valid_state())
    assert result["status"] == "pass"
    assert result["failed"] == 0
    assert result["total"] == 12


def test_missing_financial_data_fails():
    state = _valid_state()
    state["financial_data"] = None
    result = run_sanity_checks(state)
    assert result["status"] == "fail"
    assert result["checks"]["data_fetched"]["status"] == "fail"


def test_error_in_financial_data_fails():
    state = _valid_state()
    state["financial_data"] = {"error": "Network timeout", "ticker": "AAPL"}
    result = run_sanity_checks(state)
    assert result["status"] == "fail"


def test_wrong_narrative_count_fails():
    state = _valid_state()
    state["narratives"] = state["narratives"][:2]  # Only 2 narratives
    result = run_sanity_checks(state)
    assert result["checks"]["narrative_count"]["status"] == "fail"


def test_probabilities_not_summing_to_one_fails():
    state = _valid_state()
    for n in state["narratives"]:
        n["probability"] = 0.5  # Sum = 1.5
    result = run_sanity_checks(state)
    assert result["checks"]["probability_sum"]["status"] == "fail"


def test_missing_brief_fails():
    state = _valid_state()
    state["final_brief"] = None
    result = run_sanity_checks(state)
    assert result["checks"]["brief_generated"]["status"] == "fail"


def test_missing_expectations_gap_fails():
    state = _valid_state()
    state["expectations_gap"] = None
    result = run_sanity_checks(state)
    assert result["checks"]["gap_populated"]["status"] == "fail"
