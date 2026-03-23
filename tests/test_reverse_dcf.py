"""
Tests for the simplified reverse DCF (implied expectations calculation).

These verify that _compute_implied_expectations() produces sensible results
across a range of inputs — not that the math matches a Bloomberg terminal
(the model is deliberately simplified), but that the output is internally
consistent and the solver converges.
"""
import pytest
from variant.tools.yfinance_tools import _compute_implied_expectations


def test_positive_growth_company():
    """A company trading above DCF of current earnings should imply positive CAGR."""
    result = _compute_implied_expectations(
        enterprise_value=500e9,   # $500B EV
        current_revenue=100e9,    # $100B revenue
        operating_margin_pct=25.0,
    )
    assert result["implied_revenue_cagr_pct"] is not None
    assert result["implied_revenue_cagr_pct"] > 0
    assert result["method"] == "simplified_reverse_dcf"


def test_low_ev_implies_low_or_negative_growth():
    """A company with EV barely above current earnings should imply low/negative CAGR."""
    result = _compute_implied_expectations(
        enterprise_value=30e9,    # $30B EV
        current_revenue=100e9,    # $100B revenue — EV < revenue
        operating_margin_pct=10.0,
    )
    assert result["implied_revenue_cagr_pct"] is not None
    assert result["implied_revenue_cagr_pct"] < 10  # Should not imply high growth


def test_cagr_within_bounds():
    """Implied CAGR should always be within the solver's search range."""
    result = _compute_implied_expectations(
        enterprise_value=1000e9,
        current_revenue=50e9,
        operating_margin_pct=30.0,
    )
    cagr = result["implied_revenue_cagr_pct"]
    assert cagr is not None
    assert -20 <= cagr <= 80


def test_terminal_revenue_consistent_with_cagr():
    """Implied terminal revenue should equal current_revenue * (1 + CAGR)^years."""
    current_revenue = 80e9
    result = _compute_implied_expectations(
        enterprise_value=400e9,
        current_revenue=current_revenue,
        operating_margin_pct=20.0,
        projection_years=5,
    )
    cagr = result["implied_revenue_cagr_pct"] / 100.0
    years = result["assumptions"]["projection_years"]
    expected_terminal = round(current_revenue * ((1 + cagr) ** years) / 1e9, 1)
    assert result["implied_terminal_revenue_bn"] == expected_terminal


def test_assumptions_passed_through():
    """Custom assumptions should appear in the output."""
    result = _compute_implied_expectations(
        enterprise_value=200e9,
        current_revenue=50e9,
        operating_margin_pct=15.0,
        wacc=0.10,
        terminal_growth=0.025,
    )
    assert result["assumptions"]["wacc_pct"] == 10.0
    assert result["assumptions"]["terminal_growth_pct"] == 2.5
    assert result["assumptions"]["operating_margin_held_at_pct"] == 15.0


def test_higher_ev_implies_higher_growth():
    """Holding revenue/margins constant, higher EV should imply higher CAGR."""
    base = _compute_implied_expectations(
        enterprise_value=200e9,
        current_revenue=50e9,
        operating_margin_pct=20.0,
    )
    premium = _compute_implied_expectations(
        enterprise_value=400e9,
        current_revenue=50e9,
        operating_margin_pct=20.0,
    )
    assert premium["implied_revenue_cagr_pct"] > base["implied_revenue_cagr_pct"]
