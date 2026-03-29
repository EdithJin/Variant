"""
News search via DuckDuckGo.

Zero-config fallback — no API key required, just `pip install ddgs`.
Uses the ddgs library (formerly duckduckgo-search) to fetch recent news articles.

Tradeoffs vs Tavily:
- Pro: Free, no API key, no rate limits
- Con: Unofficial (can break), snippets are less LLM-optimized, no news-specific filtering
"""
import logging

logger = logging.getLogger(__name__)


def _get_ddgs_class():
    """Import DDGS from the new package name, fall back to the old one."""
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS
        return DDGS
    except ImportError:
        return None


def search_ddg(query: str, max_results: int = 5) -> list[dict]:
    """
    Search for recent news via DuckDuckGo.

    Returns results in the same format as Tavily for interchangeability.
    Returns empty list on any failure.
    """
    DDGS = _get_ddgs_class()
    if DDGS is None:
        logger.warning("ddgs not installed — skipping DDG search")
        return []

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.news(query, max_results=max_results))

        results = []
        for item in raw:
            url = item.get("url", "")
            results.append({
                "title": item.get("title", ""),
                "url": url,
                "content": item.get("body", ""),
                "published_date": item.get("date", ""),
                "source": url.split("/")[2] if url else item.get("source", ""),
            })
        return results

    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return []
