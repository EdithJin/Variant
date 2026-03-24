"""
Variant Evaluation System

Two-phase design:

  Phase 1 — ANALYSIS (daily, uses LLM):
      Run the agent on tickers from data/tickers.csv.
      Save full state + brief. Run pipeline sanity checks.
      Production runs should use Anthropic/Sonnet (LLM_PROVIDER=anthropic).

  Phase 2 — SCORING (after N days, NO LLM needed):
      Fetch current prices, compute returns, score directional accuracy.
      Score at multiple horizons: 7, 30, 90 days.
      Each scoring appends to results/scoring_log.csv (goes to GitHub).

Usage:
    # Phase 1: daily analysis (use anthropic for production)
    python -m variant.evaluate
    python -m variant.evaluate --tickers NVDA AAPL

    # Phase 2: score a past evaluation at current prices
    python -m variant.evaluate --score 2026-03-22

    # Score all past evaluations that have enough elapsed time
    python -m variant.evaluate --score-all

    # Pipeline reliability: consistency check (3 runs, same ticker)
    python -m variant.evaluate --consistency NVDA

Output:
    evaluations/YYYY-MM-DD/
    ├── analysis_summary.csv        — One row per ticker (probabilities, CAGR, gap)
    ├── sanity_summary.csv          — Pipeline sanity check results
    ├── {TICKER}_brief.txt          — Full research brief
    ├── {TICKER}_state.json         — Complete agent state
    └── {TICKER}_sanity.json        — Per-check sanity details

    results/
    └── scoring_log.csv             — Append-only scored results (goes to GitHub)
"""
import sys
import os
import csv
import json
import time
from pathlib import Path
from datetime import datetime, date, timedelta

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

from variant.graph import build_graph
from variant.tools.yfinance_tools import fetch_financial_snapshot

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "evaluations"
RESULTS_DIR = PROJECT_ROOT / "results"

# Scoring horizons: score at each of these intervals (days after analysis)
SCORING_HORIZONS = [7, 30, 90]
# Minimum days elapsed before scoring at a given horizon (allows some slack)
HORIZON_MIN_DAYS = {7: 5, 30: 25, 90: 80}


# ── Ticker basket ─────────────────────────────────────────────────────

def load_basket(tickers_csv: Path = None) -> list[dict]:
    csv_path = tickers_csv or DATA_DIR / "tickers.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    with open(csv_path) as f:
        return list(csv.DictReader(f))


# ── State helpers ─────────────────────────────────────────────────────

def _make_initial_state(ticker: str, query: str) -> dict:
    return {
        "query": query, "ticker": ticker, "narratives": [],
        "business_context": None, "financial_data": None,
        "expectations_data": None, "news_sentiment": None,
        "filings_data": None, "base_rate_data": None,
        "expectations_gap": None, "contradictions": [],
        "base_rate_flags": [], "follow_up_questions": [],
        "iteration": 0, "needs_more_data": False,
        "analyst_reasoning_summary": None, "final_brief": None,
    }


def _serialize_state(state: dict) -> dict:
    return {k: v.value if hasattr(v, "value") else v for k, v in state.items()}


# ── Pipeline sanity checks (Phase 1) ─────────────────────────────────
#
# These are REGRESSION tests, not evaluation. They verify the pipeline
# passed data correctly and produced structurally valid output.
# They do NOT assess whether the analysis is right — that's Phase 2.

def run_sanity_checks(state: dict) -> dict:
    """Verify pipeline produced structurally valid output."""
    fd = state.get("financial_data")
    checks = {}

    # Data flow: did yfinance data arrive and look reasonable?
    if not fd or fd.get("error"):
        checks["data_fetched"] = {"status": "fail", "reason": fd.get("error", "No data") if fd else "No data"}
        return _summarize_checks(checks)

    checks["data_fetched"] = {"status": "pass"}

    # Spot-check a few key fields exist and are non-null
    for field in ["current_price", "market_cap_bn", "revenue_bn_ttm"]:
        checks[f"has_{field}"] = {
            "status": "pass" if fd.get(field) is not None else "fail",
            "value": fd.get(field),
        }

    # Implied expectations computed?
    ie = fd.get("implied_expectations")
    if ie:
        cagr = ie.get("implied_revenue_cagr_pct")
        checks["implied_cagr_computed"] = {"status": "pass" if cagr is not None else "fail", "value": cagr}
        checks["implied_cagr_sane"] = {"status": "pass" if cagr is not None and -30 <= cagr <= 80 else "fail", "value": cagr}
    else:
        checks["implied_cagr_computed"] = {"status": "fail", "reason": "No implied_expectations in financial_data"}

    # Narrative structure
    narratives = state.get("narratives", [])
    checks["narrative_count"] = {"status": "pass" if len(narratives) == 3 else "fail", "value": len(narratives)}

    prob_sum = sum(n.get("probability", 0) for n in narratives)
    checks["probability_sum"] = {"status": "pass" if 0.95 <= prob_sum <= 1.05 else "fail", "value": round(prob_sum, 3)}

    labels = sorted(n.get("label", "").lower() for n in narratives)
    checks["narrative_labels"] = {"status": "pass" if labels == ["base", "bear", "bull"] else "fail", "value": labels}

    # Expectations gap populated?
    gap = state.get("expectations_gap")
    checks["gap_populated"] = {"status": "pass" if gap and gap.get("gap_assessment") else "fail"}
    checks["gap_has_closest"] = {"status": "pass" if gap and gap.get("closest_narrative") else "fail"}

    # Brief generated?
    checks["brief_generated"] = {"status": "pass" if state.get("final_brief") else "fail"}

    return _summarize_checks(checks)


def _summarize_checks(checks: dict) -> dict:
    passed = sum(1 for c in checks.values() if c["status"] == "pass")
    failed = sum(1 for c in checks.values() if c["status"] == "fail")
    return {"status": "pass" if failed == 0 else "fail", "passed": passed, "failed": failed, "total": len(checks), "checks": checks}


# ── Single run ────────────────────────────────────────────────────────

def run_single(ticker: str, query: str, graph) -> dict:
    initial_state = _make_initial_state(ticker, query)
    final_state = None
    for event in graph.stream(initial_state, stream_mode="values"):
        final_state = event
    return final_state or initial_state


# ── Phase 1: Analysis ─────────────────────────────────────────────────

def run_analysis(basket: list[dict] = None):
    basket = basket or load_basket()
    today = date.today().isoformat()
    out_dir = EVAL_DIR / today
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    model = os.environ.get("REASONING_MODEL", "default")
    print(f"Variant Evaluation — Phase 1 (Analysis)")
    print(f"Date: {today} | Provider: {provider} | Model: {model}")
    print(f"Tickers: {len(basket)} | Output: {out_dir}")
    print("-" * 55)

    if provider == "groq":
        print("⚠  Running on Groq (free tier). Production evaluations should use Anthropic/Sonnet.")
        print()

    graph = build_graph()
    analysis_rows = []
    sanity_rows = []

    for i, entry in enumerate(basket):
        ticker = entry["ticker"]
        query = entry["query"]
        sector = entry.get("sector", "")
        size = entry.get("size", "")
        print(f"[{i+1}/{len(basket)}] {ticker} ({sector}/{size})")
        start = time.time()

        try:
            state = run_single(ticker, query, graph)
            elapsed = round(time.time() - start, 1)

            # Save outputs
            (out_dir / f"{ticker}_brief.txt").write_text(state.get("final_brief", ""))
            (out_dir / f"{ticker}_state.json").write_text(json.dumps(_serialize_state(state), indent=2, default=str))

            # Pipeline sanity checks
            sanity = run_sanity_checks(state)
            (out_dir / f"{ticker}_sanity.json").write_text(json.dumps(sanity, indent=2))
            print(f"  {elapsed}s | sanity: {sanity['passed']}✓ {sanity['failed']}✗")

            # Extract row for analysis_summary.csv
            narratives = state.get("narratives", [])
            probs = {n.get("label", "?"): n.get("probability", 0) for n in narratives}
            gap = state.get("expectations_gap") or {}
            fd = state.get("financial_data") or {}

            analysis_rows.append({
                "date": today, "ticker": ticker, "sector": sector, "size": size,
                "query": query, "provider": provider, "model": model,
                "price_at_analysis": fd.get("current_price"),
                "market_cap_bn": fd.get("market_cap_bn"),
                "revenue_growth_pct": fd.get("revenue_growth_yoy_pct"),
                "implied_cagr_pct": gap.get("price_implied_growth_pct"),
                "closest_narrative": gap.get("closest_narrative"),
                "gap_assessment": (gap.get("gap_assessment") or "").replace("\n", " "),
                "prob_bull": round(probs.get("bull", 0), 3),
                "prob_base": round(probs.get("base", 0), 3),
                "prob_bear": round(probs.get("bear", 0), 3),
                "n_contradictions": len(state.get("contradictions", [])),
                "n_base_rate_flags": len(state.get("base_rate_flags", [])),
                "elapsed_s": elapsed, "status": "success",
            })
            sanity_rows.append({
                "date": today, "ticker": ticker,
                "sanity_status": sanity["status"],
                "passed": sanity["passed"], "failed": sanity["failed"],
            })

        except Exception as e:
            elapsed = round(time.time() - start, 1)
            print(f"  {elapsed}s | ERROR: {type(e).__name__}: {e}")
            analysis_rows.append({
                "date": today, "ticker": ticker, "sector": sector, "size": size,
                "query": query, "provider": provider, "model": model,
                "price_at_analysis": None, "market_cap_bn": None,
                "revenue_growth_pct": None, "implied_cagr_pct": None,
                "closest_narrative": None, "gap_assessment": str(e),
                "prob_bull": None, "prob_base": None, "prob_bear": None,
                "n_contradictions": None, "n_base_rate_flags": None,
                "elapsed_s": elapsed, "status": "error",
            })
            sanity_rows.append({
                "date": today, "ticker": ticker,
                "sanity_status": "error", "passed": 0, "failed": 0,
            })

    _write_csv(out_dir / "analysis_summary.csv", analysis_rows)
    _write_csv(out_dir / "sanity_summary.csv", sanity_rows)

    n_ok = sum(1 for r in analysis_rows if r["status"] == "success")
    n_err = sum(1 for r in analysis_rows if r["status"] == "error")
    print(f"\n{'=' * 55}")
    print(f"PHASE 1 COMPLETE — {today}")
    print(f"Success: {n_ok}/{len(basket)} | Errors: {n_err}")
    print(f"Next: score with --score {today} (after 7+ days)")
    print(f"{'=' * 55}")


# ── Phase 2: Scoring ──────────────────────────────────────────────────

def score_evaluation(eval_date: str):
    """
    Score a past evaluation at the current date.

    Directional scoring:
      closest=bull  + stock down → CORRECT (market was too optimistic)
      closest=bull  + stock up   → WRONG   (market was right)
      closest=bear  + stock up   → CORRECT (market was too pessimistic)
      closest=bear  + stock down → WRONG   (market was right)
      closest=base  or flat      → NEUTRAL

    Records days_elapsed so we can analyze accuracy by time horizon.
    """
    eval_dir = EVAL_DIR / eval_date
    summary_csv = eval_dir / "analysis_summary.csv"
    if not summary_csv.exists():
        print(f"Error: {summary_csv} not found")
        sys.exit(1)

    with open(summary_csv) as f:
        analyses = list(csv.DictReader(f))

    today = date.today()
    eval_dt = date.fromisoformat(eval_date)
    days_elapsed = (today - eval_dt).days

    print(f"Scoring evaluation from {eval_date} ({days_elapsed} days ago)")
    print("-" * 55)

    score_rows = []
    for entry in analyses:
        ticker = entry["ticker"]
        if entry.get("status") != "success":
            continue

        price_str = entry.get("price_at_analysis")
        if not price_str or price_str == "None":
            continue
        price_at_analysis = float(price_str)

        fresh = fetch_financial_snapshot(ticker)
        current_price = fresh.get("current_price")
        if not current_price or fresh.get("error"):
            print(f"  {ticker}: skip (fetch failed)")
            continue

        return_pct = round((current_price - price_at_analysis) / price_at_analysis * 100, 2)
        direction = "up" if return_pct > 1 else ("down" if return_pct < -1 else "flat")

        closest = entry.get("closest_narrative", "").lower()
        if closest == "bull":
            gap_score = "correct" if direction == "down" else ("wrong" if direction == "up" else "neutral")
        elif closest == "bear":
            gap_score = "correct" if direction == "up" else ("wrong" if direction == "down" else "neutral")
        else:
            gap_score = "neutral"

        print(f"  {ticker}: ${price_at_analysis:.0f} → ${current_price:.0f} ({return_pct:+.1f}%) | {closest} | {gap_score}")

        score_rows.append({
            "eval_date": eval_date, "score_date": today.isoformat(),
            "days_elapsed": days_elapsed,
            "ticker": ticker, "sector": entry.get("sector", ""),
            "size": entry.get("size", ""),
            "price_at_analysis": price_at_analysis,
            "price_at_scoring": current_price,
            "return_pct": return_pct, "direction": direction,
            "implied_cagr_pct": entry.get("implied_cagr_pct", ""),
            "closest_narrative": closest,
            "prob_bull": entry.get("prob_bull", ""),
            "prob_base": entry.get("prob_base", ""),
            "prob_bear": entry.get("prob_bear", ""),
            "gap_score": gap_score,
            "gap_assessment": entry.get("gap_assessment", ""),
            "provider": entry.get("provider", ""),
            "model": entry.get("model", ""),
        })

    if not score_rows:
        print("No tickers scored.")
        return

    # Save to eval directory
    _write_csv(eval_dir / f"scores_{days_elapsed}d.csv", score_rows)

    # Append to results/scoring_log.csv
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "scoring_log.csv"
    _append_csv(log_path, score_rows)

    # Summary
    n_correct = sum(1 for r in score_rows if r["gap_score"] == "correct")
    n_wrong = sum(1 for r in score_rows if r["gap_score"] == "wrong")
    n_neutral = sum(1 for r in score_rows if r["gap_score"] in ("neutral", "unknown"))
    total_scored = n_correct + n_wrong
    print(f"\n{'=' * 55}")
    print(f"SCORING — {eval_date} → {today.isoformat()} ({days_elapsed}d)")
    print(f"Correct: {n_correct} | Wrong: {n_wrong} | Neutral: {n_neutral}")
    if total_scored > 0:
        print(f"Directional accuracy: {n_correct / total_scored * 100:.0f}% ({n_correct}/{total_scored} excluding neutral)")
    print(f"Saved to: {log_path}")
    print(f"{'=' * 55}")


def score_all_evaluations():
    """Score all past evaluations that have enough elapsed time for each horizon."""
    if not EVAL_DIR.exists():
        print("No evaluations found.")
        return

    today = date.today()
    scored = set()

    # Load existing scoring log to avoid duplicate scoring
    log_path = RESULTS_DIR / "scoring_log.csv"
    if log_path.exists():
        with open(log_path) as f:
            for row in csv.DictReader(f):
                scored.add((row["eval_date"], row["days_elapsed"]))

    eval_dates = sorted(d.name for d in EVAL_DIR.iterdir() if d.is_dir() and (d / "analysis_summary.csv").exists())

    if not eval_dates:
        print("No evaluation directories with analysis_summary.csv found.")
        return

    print(f"Found {len(eval_dates)} evaluations. Checking scoring horizons...")
    print(f"Horizons: {SCORING_HORIZONS} days")
    print("-" * 55)

    for eval_date in eval_dates:
        eval_dt = date.fromisoformat(eval_date)
        days_elapsed = (today - eval_dt).days

        for horizon in SCORING_HORIZONS:
            min_days = HORIZON_MIN_DAYS.get(horizon, horizon - 5)
            if days_elapsed >= min_days:
                key = (eval_date, str(horizon))
                if key not in scored:
                    print(f"\nScoring {eval_date} at ~{horizon}d horizon ({days_elapsed}d elapsed):")
                    score_evaluation(eval_date)
                    # Mark all horizon levels as scored for this eval_date
                    # (since score_evaluation records actual days_elapsed, not horizon)
                    scored.add((eval_date, str(days_elapsed)))
                    break  # Only score once per eval_date per run

    print(f"\nDone. Results in: {log_path}")


# ── Consistency check ─────────────────────────────────────────────────

def run_consistency_check(ticker: str, n_runs: int = 3):
    query = "What does the market expect, and where might expectations be wrong?"
    print(f"Consistency check: {ticker} ({n_runs} runs)")
    graph = build_graph()
    runs = []
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}...")
        state = run_single(ticker, query, graph)
        probs = {n.get("label", f"unk{i}"): n.get("probability", 0) for n in state.get("narratives", [])}
        runs.append(probs)

    all_labels = sorted(set(l for r in runs for l in r))
    print(f"\n  {'Label':<8} {'Values':<30} {'Spread':<8} {'Stable'}")
    print(f"  {'-'*8} {'-'*30} {'-'*8} {'-'*6}")
    all_stable = True
    for label in all_labels:
        values = [r.get(label, 0) for r in runs]
        spread = max(values) - min(values)
        stable = spread <= 0.15
        if not stable:
            all_stable = False
        vals_str = ", ".join(f"{v:.0%}" for v in values)
        print(f"  {label:<8} {vals_str:<30} {spread:.0%}     {'✓' if stable else '✗'}")
    print(f"\n  Overall: {'PASS' if all_stable else 'WARN'} (threshold: ±15%)")


# ── CSV helpers ───────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--consistency" in args:
        idx = args.index("--consistency")
        ticker = args[idx + 1] if idx + 1 < len(args) else "NVDA"
        run_consistency_check(ticker)
    elif "--score-all" in args:
        score_all_evaluations()
    elif "--score" in args:
        idx = args.index("--score")
        eval_date = args[idx + 1] if idx + 1 < len(args) else None
        if not eval_date:
            print("Error: --score requires a date (e.g., --score 2026-03-22)")
            sys.exit(1)
        score_evaluation(eval_date)
    elif "--tickers" in args:
        idx = args.index("--tickers")
        ticker_list = [t.upper() for t in args[idx + 1:]]
        if not ticker_list:
            print("Error: --tickers requires at least one ticker")
            sys.exit(1)
        q = "What does the market expect, and where might expectations be wrong?"
        run_analysis([{"ticker": t, "query": q, "sector": "", "size": ""} for t in ticker_list])
    else:
        run_analysis()


if __name__ == "__main__":
    main()
