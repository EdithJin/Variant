"""
News search node — fetches recent news for the analyst.

Fallback chain: Tavily (if API key set) → DuckDuckGo (no key needed) → stub.
The pipeline never breaks regardless of which providers are available.
"""
import logging

from variant.state import AgentState
from variant.tools.tavily_search import search_tavily
from variant.tools.ddg_search import search_ddg

logger = logging.getLogger(__name__)


def _build_search_query(state: AgentState) -> str:
    """
    Build a search query from ticker, business context, and narratives.

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

    Returns news_sentiment dict with either real articles or stub fallback.
    """
    query = _build_search_query(state)
    articles, source = _search_with_fallback(query, max_results=6)

    if not articles:
        logger.info("No news results for %s — falling back to stub", state["ticker"])
        return {
            "news_sentiment": {
                "source": "stub",
            }
        }

    return {
        "news_sentiment": {
            "source": source,
            "query_used": query,
            "article_count": len(articles),
            "articles": articles,
        }
    }
