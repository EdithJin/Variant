"""
News search via Tavily API.

Tavily is a search engine optimized for LLM consumption — it returns clean
snippets rather than raw HTML. Free tier: 1,000 searches/month.

This module handles the raw API call. The node layer (nodes/news.py) handles
query construction, formatting, and fallback logic.
"""
import os
import logging

logger = logging.getLogger(__name__)


def search_tavily(query: str, max_results: int = 5) -> list[dict]:
    """
    Search for recent news articles via Tavily.

    Requires TAVILY_API_KEY in environment.
    Returns empty list on any failure (graceful degradation).
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        from tavily import TavilyClient, MissingAPIKeyError, InvalidAPIKeyError
    except ImportError:
        logger.warning("tavily-python not installed — skipping Tavily search")
        return []

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            topic="news",
        )
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "published_date": item.get("published_date", ""),
                "source": item.get("url", "").split("/")[2] if item.get("url") else "",
            })
        return results

    except (MissingAPIKeyError, InvalidAPIKeyError) as e:
        logger.warning("Tavily API key error: %s", e)
        return []
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return []
