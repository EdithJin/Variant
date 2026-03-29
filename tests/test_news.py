"""Tests for the news search node and formatting."""
from unittest.mock import patch

from variant.nodes.news import news_node, _build_search_query, _search_with_fallback
from variant.nodes.analyst import _format_news


# ── Query construction ─────────────────────────────────────────────────

def _make_state(**overrides):
    base = {
        "ticker": "NVDA",
        "query": "Is NVDA overvalued?",
        "business_context": {"company_name": "NVIDIA Corporation"},
        "narratives": [
            {
                "label": "bull",
                "story": "AI demand keeps growing",
                "probability": 0.4,
                "key_assumptions": ["Data center revenue doubles"],
                "supporting_evidence": [],
                "contradicting_evidence": [],
            }
        ],
    }
    base.update(overrides)
    return base


def test_build_search_query_includes_company_and_ticker():
    state = _make_state()
    query = _build_search_query(state)
    assert "NVIDIA Corporation" in query
    assert "NVDA" in query


def test_build_search_query_includes_narrative_assumptions():
    state = _make_state()
    query = _build_search_query(state)
    assert "Data center revenue doubles" in query


def test_build_search_query_without_narratives():
    state = _make_state(narratives=[])
    query = _build_search_query(state)
    assert "NVDA" in query


def test_build_search_query_without_business_context():
    state = _make_state(business_context=None)
    query = _build_search_query(state)
    assert "NVDA" in query


# ── Fallback chain ────────────────────────────────────────────────────

FAKE_ARTICLES = [
    {
        "title": "NVIDIA beats earnings expectations",
        "url": "https://example.com/nvidia-earnings",
        "content": "NVIDIA reported Q4 revenue of $22.1B, beating estimates.",
        "published_date": "2026-03-25",
        "source": "example.com",
    },
    {
        "title": "AI chip demand surges",
        "url": "https://example.com/ai-chips",
        "content": "Data center GPU shipments hit record highs.",
        "published_date": "2026-03-24",
        "source": "example.com",
    },
]


@patch("variant.nodes.news.search_ddg", return_value=[])
@patch("variant.nodes.news.search_tavily", return_value=FAKE_ARTICLES)
def test_fallback_uses_tavily_first(mock_tavily, mock_ddg):
    articles, source = _search_with_fallback("NVDA", 5)
    assert source == "tavily"
    assert len(articles) == 2
    mock_ddg.assert_not_called()


@patch("variant.nodes.news.search_ddg", return_value=FAKE_ARTICLES)
@patch("variant.nodes.news.search_tavily", return_value=[])
def test_fallback_uses_ddg_when_tavily_empty(mock_tavily, mock_ddg):
    articles, source = _search_with_fallback("NVDA", 5)
    assert source == "duckduckgo"
    assert len(articles) == 2


@patch("variant.nodes.news.search_ddg", return_value=[])
@patch("variant.nodes.news.search_tavily", return_value=[])
def test_fallback_returns_stub_when_all_fail(mock_tavily, mock_ddg):
    articles, source = _search_with_fallback("NVDA", 5)
    assert source == "stub"
    assert articles == []


# ── News node with mock ────────────────────────────────────────────────

@patch("variant.nodes.news._search_with_fallback", return_value=(FAKE_ARTICLES, "tavily"))
def test_news_node_returns_articles(mock_search):
    state = _make_state()
    result = news_node(state)
    ns = result["news_sentiment"]
    assert ns["source"] == "tavily"
    assert ns["article_count"] == 2
    assert len(ns["articles"]) == 2
    assert ns["articles"][0]["title"] == "NVIDIA beats earnings expectations"


@patch("variant.nodes.news._search_with_fallback", return_value=(FAKE_ARTICLES, "duckduckgo"))
def test_news_node_tracks_ddg_source(mock_search):
    state = _make_state()
    result = news_node(state)
    assert result["news_sentiment"]["source"] == "duckduckgo"


@patch("variant.nodes.news._search_with_fallback", return_value=([], "stub"))
def test_news_node_falls_back_to_stub(mock_search):
    state = _make_state()
    result = news_node(state)
    assert result["news_sentiment"]["source"] == "stub"


# ── Analyst formatting ─────────────────────────────────────────────────

def test_format_news_with_articles():
    data = {
        "source": "tavily",
        "article_count": 2,
        "articles": FAKE_ARTICLES,
    }
    formatted = _format_news(data)
    assert "NVIDIA beats earnings expectations" in formatted
    assert "AI chip demand surges" in formatted
    assert "Analytical guidance" in formatted
    assert "example.com" in formatted


def test_format_news_stub_fallback():
    formatted = _format_news({"source": "stub"})
    assert "Not available" in formatted


def test_format_news_none():
    formatted = _format_news(None)
    assert "Not available" in formatted
