"""
yfinance data extraction utilities.

Design principle: all functions return clean Python dicts with simple types
(str, float, int, None). No pandas DataFrames or yfinance objects leak outside
this module. This keeps the agent state serializable and the LLM prompts clean.

The fetch_financial_snapshot() function collects everything the analyst needs
in a single call: price data, growth rates, margins, valuation multiples,
analyst consensus, and EPS surprise history.

The fetch_market_context() function provides multi-timeframe relative
performance data (ticker vs SPY/QQQ/sector ETF + VIX) so the analyst can
separate systematic (market-wide) moves from idiosyncratic (stock-specific) ones.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def _safe_get(d: dict, key: str, default=None):
    val = d.get(key, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


def _pct(val, default=None):
    """Convert decimal fraction to percentage string."""
    if val is None:
        return default
    return round(val * 100, 1)


def _compute_implied_expectations(
    enterprise_value: float,
    current_revenue: float,
    operating_margin_pct: float,
    wacc: float = 0.09,
    terminal_growth: float = 0.03,
    tax_rate: float = 0.21,
    projection_years: int = 5,
) -> dict:
    """
    Simplified reverse DCF: given the current enterprise value, solve for the
    implied revenue CAGR over the projection period.

    Assumptions (conservative defaults):
    - WACC: 9% (typical for large-cap equities)
    - Terminal growth: 3% (GDP-ish)
    - Tax rate: 21% (US corporate)
    - Operating margin held constant at current level
    - No change in capital intensity (NOPAT ≈ operating income × (1 - tax))

    This is deliberately simplified. A full Mauboussin reverse DCF would also
    solve for implied margin trajectory and reinvestment rate. But this gives
    us the most important number: what growth does the market price imply?
    """
    op_margin = operating_margin_pct / 100.0

    # Try revenue CAGR from -20% to +80% in 1% steps, find which one
    # produces a DCF value closest to the actual enterprise value
    best_cagr = None
    best_diff = float("inf")

    for cagr_bps in range(-20_00, 80_01, 100):  # -20% to +80%
        cagr = cagr_bps / 100_00

        dcf_value = 0.0
        projected_revenue = current_revenue
        for year in range(1, projection_years + 1):
            projected_revenue *= (1 + cagr)
            nopat = projected_revenue * op_margin * (1 - tax_rate)
            dcf_value += nopat / ((1 + wacc) ** year)

        # Terminal value (Gordon growth model on final year NOPAT)
        final_nopat = projected_revenue * op_margin * (1 - tax_rate)
        terminal_value = final_nopat * (1 + terminal_growth) / (wacc - terminal_growth)
        dcf_value += terminal_value / ((1 + wacc) ** projection_years)

        diff = abs(dcf_value - enterprise_value)
        if diff < best_diff:
            best_diff = diff
            best_cagr = cagr

    implied_cagr_pct = round(best_cagr * 100, 1) if best_cagr is not None else None

    # Also compute what the terminal year revenue would be
    implied_terminal_revenue = None
    if best_cagr is not None:
        implied_terminal_revenue = round(
            current_revenue * ((1 + best_cagr) ** projection_years) / 1e9, 1
        )

    return {
        "implied_revenue_cagr_pct": implied_cagr_pct,
        "implied_terminal_revenue_bn": implied_terminal_revenue,
        "assumptions": {
            "wacc_pct": wacc * 100,
            "terminal_growth_pct": terminal_growth * 100,
            "tax_rate_pct": tax_rate * 100,
            "operating_margin_held_at_pct": operating_margin_pct,
            "projection_years": projection_years,
        },
        "method": "simplified_reverse_dcf",
    }


def fetch_business_context(ticker_symbol: str) -> dict:
    """
    Fetches ONLY the qualitative business context for a ticker — company name,
    sector, industry, and business description. No financial ratios or multiples.

    This is used by the hypothesis generator to construct narratives from
    business understanding (Damodaran's framework), without anchoring on
    valuation numbers that would bias the narrative toward confirming
    what the price already shows.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
    except Exception as e:
        return {
            "ticker": ticker_symbol,
            "error": f"Failed to fetch data: {type(e).__name__}: {e}",
            "company_name": ticker_symbol,
        }

    return {
        "ticker": ticker_symbol,
        "company_name": _safe_get(info, "longName") or ticker_symbol,
        "sector": _safe_get(info, "sector"),
        "industry": _safe_get(info, "industry"),
        "business_summary": _safe_get(info, "longBusinessSummary"),
        "full_time_employees": _safe_get(info, "fullTimeEmployees"),
        "country": _safe_get(info, "country"),
    }


def fetch_financial_snapshot(ticker_symbol: str) -> dict:
    """
    Fetches a clean financial snapshot for a ticker using yfinance.
    Returns a dict ready for the analyst prompt — no raw DataFrames.

    On failure (bad ticker, network error, API changes), returns a minimal
    dict with the ticker and an error field so downstream LLM nodes can
    still reason (with lower confidence) instead of crashing the graph.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
    except Exception as e:
        return {
            "ticker": ticker_symbol,
            "error": f"Failed to fetch data: {type(e).__name__}: {e}",
            "company_name": ticker_symbol,
        }

    # yfinance returns an empty dict or a dict with just {'trailingPegRatio': ...}
    # for invalid tickers — detect this
    if not info or not info.get("marketCap"):
        return {
            "ticker": ticker_symbol,
            "error": f"No data found for ticker '{ticker_symbol}' — verify the symbol is correct",
            "company_name": info.get("longName") or ticker_symbol,
        }

    # ── Core price & valuation ──────────────────────────────────────
    price = _safe_get(info, "currentPrice") or _safe_get(info, "regularMarketPrice")
    market_cap = _safe_get(info, "marketCap")
    market_cap_bn = round(market_cap / 1e9, 1) if market_cap else None

    # ── Growth & profitability ──────────────────────────────────────
    revenue_growth_pct = _pct(_safe_get(info, "revenueGrowth"))
    earnings_growth_pct = _pct(_safe_get(info, "earningsGrowth"))
    gross_margin_pct = _pct(_safe_get(info, "grossMargins"))
    operating_margin_pct = _pct(_safe_get(info, "operatingMargins"))
    net_margin_pct = _pct(_safe_get(info, "profitMargins"))

    # ── Valuation multiples ─────────────────────────────────────────
    forward_pe = _safe_get(info, "forwardPE")
    trailing_pe = _safe_get(info, "trailingPE")
    price_to_book = _safe_get(info, "priceToBook")
    ev_to_ebitda = _safe_get(info, "enterpriseToEbitda")
    price_to_sales = _safe_get(info, "priceToSalesTrailingTwelveMonths")

    # ── Revenue & earnings scale ────────────────────────────────────
    total_revenue = _safe_get(info, "totalRevenue")
    revenue_bn = round(total_revenue / 1e9, 2) if total_revenue else None
    ebitda = _safe_get(info, "ebitda")
    ebitda_bn = round(ebitda / 1e9, 2) if ebitda else None
    free_cashflow = _safe_get(info, "freeCashflow")
    fcf_bn = round(free_cashflow / 1e9, 2) if free_cashflow else None

    # ── 52-week range ───────────────────────────────────────────────
    high_52w = _safe_get(info, "fiftyTwoWeekHigh")
    low_52w = _safe_get(info, "fiftyTwoWeekLow")
    pct_from_high = None
    if price and high_52w:
        pct_from_high = round((price / high_52w - 1) * 100, 1)

    # ── Analyst consensus ───────────────────────────────────────────
    target_price = _safe_get(info, "targetMeanPrice")
    upside_pct = None
    if price and target_price:
        upside_pct = round((target_price / price - 1) * 100, 1)
    num_analysts = _safe_get(info, "numberOfAnalystOpinions")
    recommendation = _safe_get(info, "recommendationKey")  # "buy", "hold", etc.

    # ── EPS surprise history ────────────────────────────────────────
    eps_surprises = []
    try:
        hist_earnings = ticker.earnings_history
        if hist_earnings is not None and not hist_earnings.empty:
            for _, row in hist_earnings.head(4).iterrows():
                actual = row.get("epsActual")
                estimate = row.get("epsEstimate")
                surprise_pct = None
                if actual is not None and estimate and estimate != 0:
                    surprise_pct = round((actual - estimate) / abs(estimate) * 100, 1)
                eps_surprises.append({
                    "quarter": str(row.name) if hasattr(row, "name") else None,
                    "actual": actual,
                    "estimate": estimate,
                    "surprise_pct": surprise_pct,
                })
    except Exception:
        pass

    # ── Sector / industry context ───────────────────────────────────
    sector = _safe_get(info, "sector")
    industry = _safe_get(info, "industry")
    company_name = _safe_get(info, "longName") or ticker_symbol

    # ── Implied expectations (simplified reverse DCF) ──────────────
    # Solve for: what revenue CAGR does the current enterprise value imply,
    # given current margins and a reasonable cost of capital?
    # This is a simplified version of Mauboussin's reverse DCF.
    enterprise_value = _safe_get(info, "enterpriseValue")
    ev_bn = round(enterprise_value / 1e9, 2) if enterprise_value else None

    implied_expectations = None
    if enterprise_value and total_revenue and operating_margin_pct is not None:
        implied_expectations = _compute_implied_expectations(
            enterprise_value=enterprise_value,
            current_revenue=total_revenue,
            operating_margin_pct=operating_margin_pct,
        )

    return {
        "ticker": ticker_symbol,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "current_price": price,
        "market_cap_bn": market_cap_bn,
        "revenue_bn_ttm": revenue_bn,
        "ebitda_bn_ttm": ebitda_bn,
        "free_cash_flow_bn_ttm": fcf_bn,
        "revenue_growth_yoy_pct": revenue_growth_pct,
        "earnings_growth_yoy_pct": earnings_growth_pct,
        "gross_margin_pct": gross_margin_pct,
        "operating_margin_pct": operating_margin_pct,
        "net_margin_pct": net_margin_pct,
        "forward_pe": round(forward_pe, 1) if forward_pe else None,
        "trailing_pe": round(trailing_pe, 1) if trailing_pe else None,
        "price_to_book": round(price_to_book, 2) if price_to_book else None,
        "ev_to_ebitda": round(ev_to_ebitda, 1) if ev_to_ebitda else None,
        "price_to_sales": round(price_to_sales, 2) if price_to_sales else None,
        "price_52w_high": high_52w,
        "price_52w_low": low_52w,
        "pct_from_52w_high": pct_from_high,
        "analyst_target_price": target_price,
        "analyst_upside_pct": upside_pct,
        "analyst_consensus": recommendation,
        "num_analysts": num_analysts,
        "eps_surprise_history": eps_surprises,
        "enterprise_value_bn": ev_bn,
        "implied_expectations": implied_expectations,
    }


# ── Sector → ETF mapping ────────────────────────────────────────────────
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
}


def _compute_return(hist: pd.DataFrame, days: int) -> float | None:
    """Compute percentage return over the last N trading days from a price history."""
    if hist is None or hist.empty or len(hist) < 2:
        return None
    close = hist["Close"]
    if len(close) <= days:
        # Use whatever history we have
        start_price = float(close.iloc[0])
    else:
        start_price = float(close.iloc[-days - 1])
    end_price = float(close.iloc[-1])
    if start_price and start_price != 0:
        return round((end_price / start_price - 1) * 100, 1)
    return None


def fetch_market_context(ticker_symbol: str, sector: str | None = None) -> dict:
    """
    Fetch multi-timeframe relative performance for a ticker vs broad market
    benchmarks and its sector ETF, plus VIX as a volatility indicator.

    Returns a dict with returns at 1-day, 5-day, 1-month, and YTD timeframes
    for the ticker, SPY, QQQ, and the sector ETF. Also includes the
    relative-to-SPY returns (the key number for isolating stock-specific moves).

    All data from yfinance — no new API needed.
    """
    # Determine sector ETF
    sector_etf = SECTOR_ETF_MAP.get(sector, None) if sector else None

    # Build list of symbols to fetch
    symbols = [ticker_symbol, "SPY", "QQQ"]
    if sector_etf and sector_etf not in symbols:
        symbols.append(sector_etf)

    # Fetch enough history to cover YTD + buffer
    today = datetime.now()
    jan1 = datetime(today.year, 1, 1)
    ytd_days = (today - jan1).days + 10  # buffer for weekends/holidays
    period_days = max(ytd_days, 40)  # at least 40 days for 1-month calc

    # Timeframe definitions (in trading days)
    timeframes = {
        "1d": 1,
        "5d": 5,
        "1mo": 21,
    }

    results = {}
    histories = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{period_days}d")
            histories[symbol] = hist

            returns = {}
            for tf_name, tf_days in timeframes.items():
                returns[tf_name] = _compute_return(hist, tf_days)

            # YTD: compute from Jan 1 specifically
            if hist is not None and not hist.empty:
                jan1_date = pd.Timestamp(datetime(today.year, 1, 1), tz=hist.index.tz)
                ytd_hist = hist[hist.index >= jan1_date]
                if len(ytd_hist) >= 2:
                    returns["ytd"] = round(
                        (float(ytd_hist["Close"].iloc[-1]) / float(ytd_hist["Close"].iloc[0]) - 1) * 100, 1
                    )
                else:
                    returns["ytd"] = None
            else:
                returns["ytd"] = None

            results[symbol] = returns
        except Exception:
            results[symbol] = {"1d": None, "5d": None, "1mo": None, "ytd": None}

    # Compute relative-to-SPY returns
    ticker_returns = results.get(ticker_symbol, {})
    spy_returns = results.get("SPY", {})
    relative_to_spy = {}
    for tf in ["1d", "5d", "1mo", "ytd"]:
        t_ret = ticker_returns.get(tf)
        s_ret = spy_returns.get(tf)
        if t_ret is not None and s_ret is not None:
            relative_to_spy[tf] = round(t_ret - s_ret, 1)
        else:
            relative_to_spy[tf] = None

    # Fetch VIX level
    vix = None
    try:
        vix_ticker = yf.Ticker("^VIX")
        vix_hist = vix_ticker.history(period="2d")
        if vix_hist is not None and not vix_hist.empty:
            vix = round(float(vix_hist["Close"].iloc[-1]), 1)
    except Exception:
        pass

    return {
        "ticker": ticker_symbol,
        "sector_etf": sector_etf,
        "returns": results,
        "relative_to_spy": relative_to_spy,
        "vix": vix,
    }
