"""
News search node — fetches recent news for the analyst.

Runs two search scopes:
1. Ticker-specific — company name + ticker + narrative themes
2. Macro/market-wide — broad market news that may affect the ticker
   (e.g., geopolitical events, rate decisions, sector-wide moves)

This dual-scope approach ensures the analyst sees both company-specific
catalysts and systematic drivers that wouldn't surface in ticker-only searches.

Fallback chain per scope: Tavily (if API key set) → DuckDuckGo (no key needed) → empty.
The pipeline never breaks regardless of which providers are available.
"""
import logging

from variant.state import AgentState
from variant.tools.tavily_search import search_tavily
from variant.tools.ddg_search import search_ddg

logger = logging.getLogger(__name__)


def _build_search_query(state: AgentState) -> str:
    """
    Build a ticker-specific search query from ticker, business context, and narratives.

    Uses the company name + ticker for specificity, plus key terms from
    the narratives to surface relevant catalysts and contradictions.
    """
    ticker = state["ticker"]
    bc = state.get("business_context") or {}
    company_name = bc.get("company_name", ticker)

    # Base query: company + ticker for precision
    query = f'"{company_name}" stock {ticker}'

    # Extract key themes from narratives to focus the search
    narratives = state.get("narratives", [])
    if narratives:
        # Pull a few key assumptions across narratives for search context
        assumptions = []
        for n in narratives[:3]:
            for a in (n.get("key_assumptions") or [])[:1]:
                assumptions.append(a)
        if assumptions:
            # Take first 2 assumption snippets, truncated
            terms = " ".join(a[:60] for a in assumptions[:2])
            query += f" {terms}"

    return query


def _build_macro_query(state: AgentState) -> str:
    """
    Build a market-wide search query to surface macro/systematic drivers.

    Captures broad market events (geopolitical, rate decisions, risk-off)
    that affect individual stocks but wouldn't appear in ticker-specific searches.
    """
    bc = state.get("business_context") or {}
    sector = bc.get("sector", "")

    # Broad market query — deliberately not ticker-specific
    query = "stock market today"
    if sector:
        query += f" {sector} sector"

    return query


def _search_with_fallback(query: str, max_results: int) -> tuple[list[dict], str]:
    """
    Try each news provider in order. Returns (articles, source_name).

    Chain: Tavily → DuckDuckGo → empty.
    """
    articles = search_tavily(query, max_results)
    if articles:
        return articles, "tavily"

    articles = search_ddg(query, max_results)
    if articles:
        return articles, "duckduckgo"

    return [], "stub"


def news_node(state: AgentState) -> dict:
    """
    Fetch recent news articles relevant to the ticker being analyzed.

    Runs two search scopes:
    1. Ticker-specific (up to 6 articles) — company catalysts and developments
    2. Macro/market-wide (up to 3 articles) — systematic drivers affecting all stocks

    Returns news_sentiment dict with articles separated by scope.
    """
    # Scope 1: Ticker-specific news
    ticker_query = _build_search_query(state)
    ticker_articles, ticker_source = _search_with_fallback(ticker_query, max_results=6)

    # Scope 2: Macro/market-wide news
    macro_query = _build_macro_query(state)
    macro_articles, macro_source = _search_with_fallback(macro_query, max_results=3)

    if not ticker_articles and not macro_articles:
        logger.info("No news results for %s — falling back to stub", state["ticker"])
        return {
            "news_sentiment": {
                "source": "stub",
            }
        }

    return {
        "news_sentiment": {
            "source": ticker_source if ticker_articles else macro_source,
            "query_used": ticker_query,
            "article_count": len(ticker_articles),
            "articles": ticker_articles,
            "macro_query_used": macro_query,
            "macro_article_count": len(macro_articles),
            "macro_articles": macro_articles,
        }
    }
