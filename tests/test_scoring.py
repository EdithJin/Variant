"""
Tests for directional scoring logic.

The scoring logic encodes the core claim of expectations investing:
when the market prices in the bull narrative and the stock drops,
the agent's expectation gap assessment was correct (market was too optimistic).
"""
import pytest


def _score(closest_narrative: str, return_pct: float) -> str:
    """Replicate the scoring logic from evaluate.py."""
    direction = "up" if return_pct > 1 else ("down" if return_pct < -1 else "flat")

    if closest_narrative == "bull":
        return "correct" if direction == "down" else ("wrong" if direction == "up" else "neutral")
    elif closest_narrative == "bear":
        return "correct" if direction == "up" else ("wrong" if direction == "down" else "neutral")
    else:
        return "neutral"


# ── Bull narrative (market prices optimism) ──────────────────────────

def test_bull_priced_stock_drops_is_correct():
    """Market was too optimistic, stock fell. Agent was right."""
    assert _score("bull", -5.0) == "correct"


def test_bull_priced_stock_rises_is_wrong():
    """Market was optimistic and stock confirmed it. Agent was wrong."""
    assert _score("bull", 5.0) == "wrong"


def test_bull_priced_stock_flat_is_neutral():
    """Stock didn't move enough to score."""
    assert _score("bull", 0.5) == "neutral"


# ── Bear narrative (market prices pessimism) ─────────────────────────

def test_bear_priced_stock_rises_is_correct():
    """Market was too pessimistic, stock rose. Agent was right."""
    assert _score("bear", 5.0) == "correct"


def test_bear_priced_stock_drops_is_wrong():
    """Market was pessimistic and stock confirmed it. Agent was wrong."""
    assert _score("bear", -5.0) == "wrong"


def test_bear_priced_stock_flat_is_neutral():
    assert _score("bear", 0.3) == "neutral"


# ── Base narrative (no strong directional claim) ─────────────────────

def test_base_priced_always_neutral():
    """When market prices the base case, no directional call is made."""
    assert _score("base", 10.0) == "neutral"
    assert _score("base", -10.0) == "neutral"
    assert _score("base", 0.0) == "neutral"


# ── Edge cases ───────────────────────────────────────────────────────

def test_threshold_boundary():
    """Returns exactly at +/-1% should be neutral (flat)."""
    assert _score("bull", 1.0) == "neutral"
    assert _score("bull", -1.0) == "neutral"
    assert _score("bull", 1.01) == "wrong"   # Just over threshold
    assert _score("bull", -1.01) == "correct" # Just over threshold


def test_unknown_narrative_is_neutral():
    """Unknown or empty narrative should score as neutral."""
    assert _score("", 5.0) == "neutral"
    assert _score("unknown", -5.0) == "neutral"
