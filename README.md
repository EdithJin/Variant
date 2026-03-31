# Variant

![CI](https://github.com/EdithJin/Variant/actions/workflows/ci.yml/badge.svg)

An AI financial research agent that implements expectations investing analysis — constructing competing narratives about a company's future, reverse-engineering what the market prices in, and identifying where consensus might be wrong.

## What This Is

Most financial AI tools gather data and summarize it. Variant asks a different question: **is the market right about what it's pricing in?**

The system implements frameworks from institutional finance:

- **Expectations Investing** (Mauboussin & Rappaport) — Reverse-engineer what growth rate, margins, and reinvestment the current stock price implies. Then assess whether those expectations will be revised.
- **Narrative + Numbers** (Damodaran) — Every valuation is a story converted to numbers. Construct competing narratives (bull/base/bear), test each for plausibility, assign probability weights.
- **Base Rate Thinking** (Mauboussin/Kahneman) — Before analyzing a specific company, know what's historically typical. Flag when implied assumptions are statistically unusual.

## How It Works

Variant is a LangGraph state machine with five nodes and one conditional loop:

```
business_context → hypothesis_generator → data_gathering → analyst → synthesis
                                                ↑              |
                                                └── loop (max 3 iterations)
```

| Node | What It Does | LLM? |
|------|-------------|------|
| **Business Context** | Fetch company name, sector, industry, description. No financial ratios — qualitative only. | No |
| **Hypothesis Generator** | Construct 3 competing narratives (bull/base/bear) from business understanding alone. No price anchoring. | Yes |
| **Data Gathering** | Fetch financial snapshot via yfinance (price, margins, multiples, EPS surprises) + simplified reverse DCF. Run stub agents for news, filings, base rates. | No |
| **Analyst** | 5-step reasoning: evaluate narratives against data, find expectations gap, check base rates, detect contradictions, decide if more data needed. | Yes |
| **Synthesis** | Format the brief. Python builds ~90% (structured data). LLM writes only the executive summary and confidence assessment. | Partial |

**Key design choice: hypothesis-first.** The hypothesis generator sees only qualitative business context — no P/E ratios, no price, no margins. This prevents anchoring bias. Financial data is gathered *after* narratives are formed, to test them rather than confirm them.

**Expectations gap analysis:** A simplified reverse DCF solves for the implied revenue CAGR the current enterprise value requires, given current margins and conservative WACC/terminal growth assumptions. This produces a falsifiable number ("the market prices in 22% revenue CAGR for 5 years") that can be compared against each narrative.

## Current Status

**POC (v0.1)** — The core pipeline works end-to-end with real financial data.

What's real:
- Financial data via yfinance — price, ratios, margins, analyst consensus, EPS surprises
- Implied expectations via simplified reverse DCF
- LLM-driven narrative construction and 5-step analytical reasoning
- Structured research brief output
- Evaluation framework with pipeline sanity checks and directional scoring

What's stubbed (Phase 2):
- Market expectations data (will use FMP/Zacks for consensus estimates)
- SEC filings (will use EDGAR API + ChromaDB RAG)
- Historical base rate data (will use Damodaran's datasets)

What's live:
- News search via Tavily (primary) + DuckDuckGo (zero-config fallback)

## Evaluation

Two-phase evaluation design:

**Phase 1 — Analysis** (daily, uses LLM): Run the agent on a 30-ticker basket diversified across 8 sectors, 3 market-cap buckets, and growth/value/defensive/volatile styles. Pipeline sanity checks (12 per ticker) verify structural validity — these are regression tests, not analytical evaluation.

**Phase 2 — Scoring** (after N days, no LLM): Fetch current prices, compute returns, score whether the expectations gap assessment was directionally correct. Score at T+1 (post-earnings), 30-day, and 90-day horizons.

```bash
# Phase 1: daily analysis
python -m variant.evaluate                       # Full 30-ticker basket
python -m variant.evaluate --tickers NVDA AAPL   # Specific tickers

# Phase 2: score past evaluations
python -m variant.evaluate --score 2026-03-22    # Score one date
python -m variant.evaluate --score-all           # Score all eligible dates

# Pipeline reliability
python -m variant.evaluate --consistency NVDA    # Same ticker 3x, check spread
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# Edit .env — add GROQ_API_KEY (free) or ANTHROPIC_API_KEY
```

## Usage

```bash
python -m variant.main AAPL "Is Apple overvalued?"
python -m variant.main NVDA "Why did NVDA drop after earnings?"
python -m variant.main TSLA "What are the biggest risks for Tesla?"
```

## Project Structure

```
variant/
├── main.py                  # CLI entry point
├── config.py                # LLM provider config (Anthropic/Groq)
├── state.py                 # AgentState schema, Narrative type
├── graph.py                 # LangGraph assembly — nodes, edges, loop logic
├── evaluate.py              # Evaluation runner + directional scoring
├── nodes/
│   ├── financial_data.py    # business_context_node + financial_data_node
│   ├── hypothesis_generator.py  # LLM: construct competing narratives
│   ├── analyst.py           # LLM: 5-step analytical reasoning
│   ├── synthesis.py         # Python formatting + LLM for exec summary
│   ├── news.py              # News search node (Tavily → DuckDuckGo → stub)
│   └── stubs.py             # Placeholder data nodes (Phase 2 replacements)
├── prompts/
│   ├── hypothesis_generator.py  # Narrative construction prompts
│   ├── analyst.py               # 5-step analysis prompts
│   └── synthesis.py             # Brief synthesis prompts
└── tools/
    ├── yfinance_tools.py    # yfinance data + reverse DCF calculation
    ├── tavily_search.py     # Tavily news search (LLM-optimized snippets)
    └── ddg_search.py        # DuckDuckGo fallback (zero-config, no API key)
```

## Examples

See [examples/](examples/) for full research briefs generated by Variant:

- [NVDA](examples/NVDA.txt) — AI infrastructure, 21% implied CAGR, macro decomposition shows selloff is VIX-driven not fundamental
- [TSLA](examples/TSLA.txt) — 80% implied CAGR vs -3% actual growth, market prices beyond even the bull case
- [META](examples/META.txt) — 7% implied CAGR vs 24% actual growth, market pricing near-bear case despite strong execution
- [XOM](examples/XOM.txt) — Geopolitical re-rating, analysts say overvalued but Hormuz disruption rewrites the base case

## Configuration

Set in `.env`:

| Variable | Options | Default |
|----------|---------|---------|
| `LLM_PROVIDER` | `groq` (free), `anthropic` | `anthropic` |
| `TAVILY_API_KEY` | API key from [app.tavily.com](https://app.tavily.com) | Falls back to DuckDuckGo |

## Architecture

For a detailed walkthrough of the graph topology, node-by-node behavior, evaluation design, and Phase 2 plans, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## References

1. Mauboussin & Rappaport. *Expectations Investing.* Columbia University Press, 2021.
2. Damodaran. *Narrative and Numbers.* Columbia University Press, 2017.
3. Mauboussin. *The Base Rate Book.* Counterpoint Global / Morgan Stanley.

## License

MIT — see [LICENSE](LICENSE).
