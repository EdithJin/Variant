# Architecture

This document explains how Variant works — the graph topology, data flow, reasoning framework, and the design vision for how the system evolves from a fixed pipeline (POC) to a genuinely agentic system (Phase 2+).

---

## 1. Overview

### Current Architecture (POC)

Variant is a **LangGraph state machine** with a fixed pipeline and one conditional loop. Five nodes run in sequence. The analyst can loop back to re-gather data, but in practice it doesn't (stubs return the same data on re-run).

```
                    ┌─────────────────┐
                    │ business_       │  Fetch company name, sector,
                    │ context         │  industry, description (NO financials)
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  hypothesis_    │  LLM constructs 3 competing
                    │  generator      │  narratives (bull/base/bear)
                    └────────┬────────┘  from business understanding only
                             │
                             ▼
              ┌────▶┌─────────────────┐
              │     │ data_gathering  │  Fetch financial snapshot +
              │     │                 │  run all data sub-agents
              │     └────────┬────────┘
              │              │
              │              ▼
              │     ┌─────────────────┐
              │     │  analyst        │  5-step analytical reasoning:
              │     │                 │  evaluate narratives, expectations
              │     │                 │  gap, base rates, contradictions
              │     └────────┬────────┘
              │              │
              │         ┌────┴────┐
              │         │ needs   │
              └── Yes ──│ more    │
           (max 3 iter) │ data?   │
                        └────┬────┘
                             │ No
                             ▼
                    ┌─────────────────┐
                    │  synthesis      │  Build structured research brief
                    └─────────────────┘  (Python formatting + LLM for
                                          exec summary & confidence only)
```

### What This Architecture Is (and Isn't)

**What it is:** A hypothesis-driven pipeline where narratives are formed before data is seen, financial data tests those narratives, and the output acknowledges uncertainty with probability weights.

**What it isn't (yet):** A genuinely agentic system. The analyst receives a pre-gathered data dump and reasons about it. It can't decide at runtime to "look up what Microsoft said about capex" or "check AMD's margins for comparison." The data it sees is fixed by the pipeline, not by its own judgment. The Phase 2 architecture (Section 7) addresses this.

---

## 2. Concepts: Tools, Sub-agents, and Skills

Three distinct types of components, each with a different role:

### Tools (deterministic, no LLM)

Functions that take inputs and return outputs. No reasoning. Should be fast, cacheable, and reliable.

| Tool | What it does | Status |
|------|-------------|--------|
| `fetch_business_context(ticker)` | Company name, sector, industry, description | Real |
| `fetch_financial_snapshot(ticker)` | Price, margins, multiples, consensus, implied expectations | Real |
| `compute_implied_expectations(ev, revenue, margin)` | Simplified reverse DCF → implied revenue CAGR | Real |
| `fetch_market_context(ticker)` | Multi-timeframe relative performance vs SPY/QQQ/sector ETF + VIX | Phase 2 |
| `search_news(query)` | Web search for recent financial news (ticker-specific AND macro) | Phase 2 |
| `fetch_sec_filing(ticker, type)` | Download and summarize SEC filings | Phase 2 |
| `lookup_base_rates(sector, metric)` | Historical growth/margin/valuation percentiles | Phase 2 |

Tools are pure data retrieval. The LLM never runs inside a tool.

### Sub-agents (LLM reasoning, bounded scope)

LLM-powered components that receive context, reason about it, and produce structured output. Each sub-agent has a specific job and a Pydantic schema that constrains its output.

| Sub-agent | Input | Output | LLM role |
|-----------|-------|--------|----------|
| Hypothesis Generator | Business context + query | 3 narratives with probabilities | Construct competing stories about the business |
| Analyst | Narratives + all gathered data | Updated narratives, expectations gap, contradictions, flags | 5-step analytical reasoning |
| Synthesis | Analyst output + state | Executive summary + confidence level | Synthesize findings into 2 sentences + judgment |

Sub-agents are where the reasoning happens. Each has a system prompt defining its analytical framework and a Pydantic schema enforcing its output structure.

### Skills (modular capability packages, Phase 2+)

Skills are the answer to: "How do we give the analyst deep domain expertise without stuffing everything into the system prompt?"

Inspired by the [Agent Skills architecture](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills), skills are modular packages of instructions + tools + reference data that load on-demand:

| Skill | When activated | What it provides |
|-------|---------------|-----------------|
| Expectations Investing | Analyst processes implied expectations data | Detailed instructions for interpreting reverse DCF, comparing implied CAGR to narratives, identifying variant perception |
| Base Rate Analysis | Analyst checks growth/margin sustainability | Instructions for interpreting growth persistence percentiles, margin reversion patterns, with reference to Damodaran's empirical data |
| Earnings Reaction | Query involves earnings event | Instructions for analyzing expectations vs. actuals, guidance language interpretation, options-implied move analysis |
| Macro Decomposition | Analyst receives market context data | Instructions for separating systematic (market/sector-wide) moves from idiosyncratic (stock-specific) moves, adjusting expectations gap interpretation during macro-driven regimes |
| Competitive Analysis | Analyst fetches data for a second ticker | Instructions for relative valuation, competitive dynamics framework, market share trend analysis |

**Implementation in LangGraph (no Claude Agent SDK required):** Skills translate to dynamic prompt assembly — the analyst's system prompt stays lean, and each tool's return value includes relevant analytical instructions alongside the data. The skill's "instructions" load only when the tool is called. This achieves progressive disclosure without the full Agent Skills filesystem architecture.

---

## 3. What "Stub" Means — POC vs Phase 2

A **stub** is a function that returns structurally valid but minimal/derived data so downstream nodes always receive the expected state keys. Stubs serve three purposes:

1. **No KeyErrors** — The analyst always gets all expected fields
2. **Contract documentation** — Each stub's return shape defines what the real implementation must provide
3. **Drop-in replacement** — Swapping a stub for a real agent requires zero changes downstream

### POC vs Phase 2 per data source

| Data Source | POC (stub) | Phase 2 (real) | What changes |
|-------------|-----------|---------------|-------------|
| **Financial Data** | Real — yfinance snapshot with ~30 fields + reverse DCF | Same, possibly enriched with FMP data | Nothing |
| **Market Context** | Not implemented. No relative performance or macro data. | `fetch_market_context`: multi-timeframe (1d/5d/1mo/YTD) returns for ticker vs SPY/QQQ/sector ETF + VIX. All yfinance, no new API. | Enables systematic vs. idiosyncratic decomposition |
| **Expectations** | Derives P/E compression from yfinance. No independent consensus data. | FMP or Zacks API for consensus EPS estimates, estimate revision history, forward revenue estimates | The data that makes expectations gap rigorous |
| **News & Sentiment** | Returns `{"source": "stub"}`. Analyst sees "Not available." | Tavily/Serper web search — **dual scope**: ticker-specific AND macro/market-wide queries | Enables contradiction detection with real catalysts + macro context |
| **SEC Filings** | Returns `{"source": "stub"}` | EDGAR API for 10-K/10-Q, extract risk factors, management discussion, insider transactions. ChromaDB for RAG over long documents. | Enables risk analysis with primary sources |
| **Base Rates** | Hardcoded first-principles rules ("fewer than 10% of large-caps sustain 40%+ growth") | Damodaran's annual datasets: actual percentiles for growth persistence, margin reversion, valuation outcomes by sector/size cohort | Replaces heuristics with empirical data |

---

## 4. Context and Memory Management

### Current state: none

Every run is completely stateless. No conversation memory, no cross-run caching, no token budget management. Each run starts from scratch.

### Three problems to solve

**Problem A: Token budget management (critical for Phase 2)**

The analyst prompt currently receives ~2-3K tokens of data. When we add real news (5-10 articles), SEC filings (10-K is 50-100 pages), and base rate tables, this will blow past any reasonable budget.

Strategy:
- **Summarize-then-analyze**: Each data tool produces a condensed summary, not raw data. This is what professional analysts do — you read the analyst summary, not the full 10-K.
- **RAG for long documents**: SEC filings go into ChromaDB. The analyst queries for relevant sections based on what its narratives need, not the whole document.
- **Token budgeting**: Each data source gets a max token allocation. Financial data: ~800 tokens. News: ~600 tokens. Filings summary: ~400 tokens. Total stays predictable.

**Problem B: Cross-run caching**

Analyzing NVDA at 9am and again at 2pm hits yfinance twice for the same data.

Strategy:
- Disk-based cache keyed on `(ticker, data_type, date)`
- TTLs: price data ~15 min, fundamentals ~24h, business context ~1 week, SEC filings ~until next filing
- Saves API calls, reduces latency, prevents rate limiting

**Problem C: Multi-turn conversation memory (Phase 3)**

Natural usage is conversational: "Analyze NVDA" → "What if margins compress 5%?" → "Compare to AMD"

Strategy:
- Keep previous `AgentState` in session memory
- Follow-up queries modify or extend the existing state
- Decision logic: which nodes need to re-run? (just analyst? whole pipeline?)
- Not needed for POC demos — single-shot is sufficient

---

## 5. Node-by-Node Walkthrough

### 1. Business Context (`nodes/financial_data.py:business_context_node`)

**Purpose:** Fetch ONLY qualitative business context — company name, sector, industry, and business description. No financial ratios, no multiples, no price data.

**Why qualitative only?** This follows Damodaran's Narrative + Numbers framework. The hypothesis generator constructs narratives from *business understanding* (what does the company do, how does it compete, what industry trends matter) rather than from financial ratios that would anchor the LLM on what the price already reflects. Financial numbers come later, to *test* the narratives.

**Data source:** yfinance `ticker.info` — only the qualitative fields.

### 2. Hypothesis Generator (`nodes/hypothesis_generator.py`)

**Purpose:** Construct 3 competing narratives (bull, base, bear) BEFORE seeing financial data.

**Why hypothesis-first?** Forming hypotheses before seeing numbers prevents confirmation bias. The LLM isn't rationalizing a conclusion — it's constructing competing stories about the business that will be tested.

**What it receives:** Only the business context (name, sector, industry, description) and the user's query. No P/E ratios, no price, no margins.

**What it produces:** Three `Narrative` objects, each with:
- A 2-3 sentence story about how the business evolves
- A probability weight (must sum to 1.0)
- Key assumptions that must hold for the narrative to play out
- Empty evidence arrays (filled by the Analyst later)

**Query adaptation:** The system prompt instructs the LLM to adapt its narratives to the specific question. A risk question gets risk-framed narratives; a valuation question gets cash-flow-framed narratives. No keyword classification needed — the LLM reads the query and adapts naturally.

**LLM call:** Uses structured output (Pydantic schema → `NarrativesOutput`) to guarantee valid JSON with exactly 3 narratives.

### 3. Data Gathering (`graph.py:data_gathering_node`)

**Purpose:** Run all data sub-agents and merge their results into state. This is the first time the system sees financial numbers — AFTER hypotheses are formed.

**Current implementation (POC):**
- `financial_data_node` — Real yfinance data: price, margins, multiples, consensus, EPS surprises, and **implied expectations** (simplified reverse DCF)
- `expectations_stub_node` — Derives P/E compression signal from yfinance data
- `news_stub_node` — Placeholder
- `filings_stub_node` — Placeholder
- `base_rate_stub_node` — First-principles flags (e.g., "fewer than 10% of large-caps sustain 40%+ growth")

**Efficiency:** On loop iterations, financial data is not re-fetched (it won't change between iterations).

**Phase 2 evolution:** This node gets replaced entirely — see Section 7 (Phase 2 Architecture).

### 4. Analyst Agent (`nodes/analyst.py`)

**Purpose:** The core intelligence. Implements a 5-step analytical framework combining Mauboussin and Damodaran.

**The 5 steps:**

| Step | What It Does | Framework |
|------|-------------|-----------|
| 1. Narrative Evaluation | Review each narrative against data, apply 3P test, update probabilities | Damodaran |
| 2. Expectations Gap | Use reverse DCF to identify implied revenue CAGR, compare to narratives | Mauboussin |
| 3. Base Rate Check | Flag statistically unusual assumptions | Mauboussin/Kahneman |
| 4. Contradiction Detection | Find data points that conflict (e.g., EPS beat + stock drop) | Original |
| 5. Loop Decision | Decide whether more data would meaningfully change the analysis | Original |

**Expectations Gap (Step 2):** The financial data includes an `implied_expectations` section computed via simplified reverse DCF: given the current enterprise value, what revenue CAGR does the market price require over the next 5 years, holding current margins constant? The analyst compares this implied CAGR to the narratives to identify where market expectations may be wrong.

**Loop criteria (Step 5):** The analyst sets `needs_more_data = true` only if ALL of:
- `iteration < 3` (hard cap)
- There's a SPECIFIC factual question that better data would answer
- That answer would shift a narrative's probability by >10 percentage points

**Data presentation:** Financial data is presented as formatted readable lines, not raw JSON dumps. This reduces token waste and produces better analytical output.

**LLM call:** Uses structured output (`AnalystOutput` Pydantic schema) to ensure all required fields.

### 5. Synthesis Agent (`nodes/synthesis.py`)

**Purpose:** Format the completed analysis into a structured research brief.

**Key design decision:** The brief structure is fixed, so Python builds ~90% of it. The LLM is called ONLY for executive summary and confidence assessment — the two parts requiring judgment.

**Output structure:**
```
VARIANT RESEARCH BRIEF: [TICKER]
├── Executive Summary (LLM-generated)
├── Competing Narratives (Python-formatted from analyst output)
│   ├── Bull Case (X%): story, assumptions, evidence
│   ├── Base Case (Y%): story, assumptions, evidence
│   └── Bear Case (Z%): story, assumptions, evidence
├── Expectations Analysis (Python-formatted from reverse DCF)
├── Base Rate Check (Python-formatted from flags)
├── Key Contradictions (Python-formatted)
├── What We Don't Know (Python-formatted)
└── Data & Confidence (LLM-generated confidence level)
```

---

## 6. Implied Expectations: Simplified Reverse DCF

The `yfinance_tools.py` module includes a `_compute_implied_expectations()` function implementing a simplified version of Mauboussin's reverse DCF:

**What it solves for:** Given the current enterprise value, what revenue CAGR over the next 5 years justifies the current price, assuming current operating margins are held constant?

**Method:** Iterates over revenue CAGR candidates from -20% to +80%, computing a DCF value for each (NOPAT discounted at WACC + Gordon growth terminal value), and finds the CAGR that produces a DCF closest to the actual enterprise value.

**Default assumptions:**
- WACC: 9% (typical large-cap equity)
- Terminal growth: 3% (GDP-ish)
- Tax rate: 21% (US corporate)
- Operating margin: held at current level
- Projection period: 5 years

**Limitations (stated honestly):** A full Mauboussin reverse DCF would also solve for implied margin trajectory and reinvestment rate. This simplified version holds margins constant and solves only for revenue growth. This is the most important single variable, but it's not the complete picture.

**Output example:**
```
Implied expectations (reverse DCF): Market prices in 18.5% revenue CAGR over 5yr
  → Implies terminal revenue of $245.3B (vs current $130.5B)
  → Assumes: WACC 9.0%, terminal growth 3.0%, margin held at 32.5%
```

---

## 7. Market Context: Systematic vs. Idiosyncratic Decomposition

### The problem

A stock's price movement reflects two distinct forces: **systematic** (market-wide or sector-wide) moves driven by macro events, and **idiosyncratic** (stock-specific) moves driven by company fundamentals. Without separating these, the analyst can't interpret price action correctly.

Example: GOOG drops 8% during a geopolitical crisis. SPY drops 6%, QQQ drops 7%. The GOOG-specific component is only ~1-2% — the rest is the market repricing risk broadly. But ticker-specific news searches return nothing about GOOG declining, because the cause is macro, not company-specific. Without market context, the analyst might wrongly conclude "the market is repricing GOOG's fundamentals" when it's repricing everything.

This matters for the expectations gap analysis. The reverse DCF tells you what growth the price implies — but if the price just dropped 8% due to a war, the implied CAGR shifted for reasons unrelated to anyone's view of GOOG's revenue trajectory. The analyst needs to know this.

### Solution: `fetch_market_context(ticker)` tool

A lightweight tool that fetches multi-timeframe returns for the ticker, broad market, and sector, all from yfinance (no new API needed).

**Benchmarks fetched:**
- **SPY** — broad market (systematic risk)
- **QQQ** — tech-heavy / growth proxy
- **Sector ETF** — sector-specific moves (XLK for tech, XLF for financials, XLE for energy, etc.)

**Timeframes fetched (all in a single call):**

| Timeframe | What it captures | Why it matters |
|-----------|-----------------|----------------|
| **1-day** | Acute events (earnings, geopolitical shock) | If GOOG -4% and SPY -3.5%, today's move is systematic. If GOOG -4% and SPY flat, it's idiosyncratic. |
| **5-day** | Recent event reactions | A war that started 3 days ago shows here. Captures "this week" context. |
| **1-month** | Medium-term trend | Separates sustained selloffs from one-day blips. |
| **YTD** | Regime context | "GOOG -15% YTD, SPY -12%" is a different story than "GOOG -15% YTD, SPY +5%." |

All four timeframes are fetched together — they're essentially free (a few yfinance calls) and cost ~100 tokens in the prompt.

**Output format (presented to analyst as readable text):**

```
Market Context for GOOG (2026-03-29):
                    1-day    5-day    1-month    YTD
  GOOG              -1.2%    -4.8%    -12.3%    +2.1%
  SPY               -1.0%    -3.9%     -8.1%    -5.2%
  QQQ               -1.3%    -4.5%     -9.7%    -3.8%
  XLK (sector ETF)  -1.1%    -4.2%     -9.0%    -4.1%

  Relative to SPY:  -0.2%    -0.9%     -4.2%    +7.3%
  VIX: 28.5 (elevated — above 20 signals market stress)
```

The **"Relative to SPY"** row is the key number — it isolates the stock-specific component by subtracting the broad market move.

### How the analyst uses this

The market context data informs multiple steps of the analyst's reasoning:

**Step 2 (Expectations Gap):** Before interpreting the implied CAGR, check whether the stock is moving with the market. If relative performance is near zero across timeframes, the price move is macro-driven and the expectations gap reflects market-wide sentiment, not a changed view on the company's fundamentals. The analyst should note this and weight the gap assessment accordingly.

**Step 4 (Contradiction Detection):** Market context enables a new class of contradictions: "stock down 8% but relative to sector only down 1% — no company-specific deterioration despite the headline move" or "stock flat while sector rallied 5% — underperformance suggests stock-specific headwind even in a favorable macro."

**Narrative Construction (Hypothesis Generator):** In Phase 2, the hypothesis generator could receive a brief macro summary (e.g., "market down 8% this month, VIX elevated at 28") to ensure narratives account for the macro regime rather than constructing purely company-specific stories during a market-wide event.

### News search: ticker-specific AND macro

When the `search_news` tool is implemented (Phase 2b), it should run **two** search scopes:

1. **Ticker-specific:** `"GOOG earnings"`, `"Google AI revenue"` — surfaces company-specific catalysts
2. **Macro/market-wide:** `"stock market today"`, `"market selloff"`, `"geopolitical risk"` — surfaces systematic drivers that affect the ticker but wouldn't appear in ticker-specific searches

Both are passed to the analyst as separate context sections so it can attribute price action to the right cause.

---

## 8. Phase 2 Architecture: Analyst-as-Tool-User

The POC's biggest architectural limitation: the analyst receives a pre-gathered data dump. It can't investigate further. Phase 2 collapses `data_gathering` + `analyst` into a single agentic node.

### Current (POC): Fixed pipeline

```
data_gathering (runs ALL tools unconditionally) → analyst (reasons about dump)
```

The analyst can't say "let me check AMD's margins" or "what did management say about capex on the earnings call?" It works with whatever the pipeline gives it.

### Phase 2: Analyst with tools

```
business_context → hypothesis_generator → analyst_with_tools → synthesis
                                              ↑       │
                                              └──loop──┘
```

The analyst becomes a ReAct-style agent that calls tools dynamically during its reasoning:

```
Analyst: "The bull narrative assumes data center demand persists.
  Let me check the financials..."
  → [calls fetch_financials("NVDA")]
  → "80% revenue growth. Now let me check if this is historically sustainable..."
  → [calls lookup_base_rates("Technology", "revenue_growth_persistence")]
  → "Only 6% of companies this size sustained >40% growth for 3+ years."
  → "The bear narrative assumes AMD competition. Let me check..."
  → [calls fetch_financials("AMD")]
  → "AMD growing 35%, gaining share. I have enough. Let me synthesize."
```

### Implementation: LangGraph tool calling

```python
from langchain_core.tools import tool

@tool
def fetch_financials(ticker: str) -> str:
    """Fetch financial snapshot for a stock ticker."""
    ...

@tool
def fetch_market_context(ticker: str) -> str:
    """Fetch multi-timeframe relative performance vs market, sector, and VIX."""
    ...

@tool
def search_news(query: str, max_results: int = 5) -> str:
    """Search recent financial news and earnings commentary."""
    ...

@tool
def lookup_base_rates(sector: str, metric: str) -> str:
    """Look up historical base rates for growth persistence or margin reversion."""
    ...

# Bind tools to LLM — works with Claude, Llama, any LangChain model
analyst_llm = get_llm().bind_tools([fetch_financials, fetch_market_context, search_news, lookup_base_rates])
```

LangGraph's `ToolNode` handles the ReAct loop natively. No Claude Agent SDK required. The system stays model-agnostic.

### Why this matters

- **Genuinely agentic** — the LLM decides what data it needs based on the narratives it's testing
- **No wasted API calls** — don't fetch SEC filings if the question is about short-term earnings
- **The loop becomes meaningful** — the analyst naturally iterates (call tool → reason → call another tool) instead of the artificial "loop back to data_gathering" edge
- **Easier to extend** — adding a new tool is one function + one `@tool` decorator, not a new node + graph edge

### Progressive prompt loading (Skills pattern)

Each tool's return value includes **analytical instructions** alongside the data. When the analyst calls `lookup_base_rates`, it gets back not just the numbers but also context on how to interpret them:

```
Growth persistence data for Technology sector (>$100B market cap):
- 5yr sustained >40% growth: 6% of companies
- 5yr sustained >20% growth: 22% of companies
- Median growth rate mean-reverts to 12% within 3 years

ANALYTICAL GUIDANCE: Compare the implied CAGR from the reverse DCF to
these base rates. If the market prices in growth above the 75th
percentile for this cohort, flag it as statistically unusual.
```

This achieves progressive disclosure — the analyst only receives detailed analytical instructions for the tools it actually calls, keeping the base system prompt lean.

---

## 9. Evaluation System

### Two phases, different purposes

| Phase | When | Uses LLM? | What it tests |
|-------|------|-----------|---------------|
| **Phase 1 — Analysis** | Daily | Yes (Sonnet) | Produces analysis + pipeline sanity checks (regression tests) |
| **Phase 2 — Scoring** | After 7/30/90 days | No | Was the expectations gap directionally correct? (analytical quality) |

Phase 1 sanity checks are **regression tests** — they verify the pipeline passed data correctly and produced structurally valid output. They do NOT assess whether the analysis is right. Phase 2 scoring is the **real evaluation**.

Production evaluations must use Anthropic/Sonnet (`LLM_PROVIDER=anthropic`). Groq/Llama is for testing the pipeline only.

```bash
# Phase 1: daily analysis
python -m variant.evaluate                          # Full 30-ticker basket
python -m variant.evaluate --tickers NVDA AAPL      # Specific tickers

# Phase 2: score past evaluations
python -m variant.evaluate --score 2026-03-22       # Score one date
python -m variant.evaluate --score-all              # Score all eligible dates

# Pipeline reliability
python -m variant.evaluate --consistency NVDA        # Same ticker 3×
```

### Directional scoring logic

The agent's core claim: "I can identify where market expectations might be wrong." The test:

| Market pricing | Stock moves | Score |
|---------------|-------------|-------|
| `closest_narrative = bull` (optimism) | Down | **Correct** — market was too optimistic |
| `closest_narrative = bull` | Up | **Wrong** — market was right |
| `closest_narrative = bear` (pessimism) | Up | **Correct** — market was too pessimistic |
| `closest_narrative = bear` | Down | **Wrong** — market was right |
| `closest_narrative = base` or flat | Any | **Neutral** |

Random would score ~50%. Consistently beating 50% is genuinely meaningful.

### Scoring horizons

Score the same analysis at multiple time intervals. The analysis is recorded once; scoring is just a price lookup (free, no LLM).

| Horizon | What it captures | Min days to trigger |
|---------|-----------------|-------------------|
| **7 days** | Short-term repricing, post-earnings reactions | 5 |
| **30 days** | Medium-term mispricing correction | 25 |
| **90 days** | Fundamental thesis playing out | 80 |

Each scoring records `days_elapsed` in the output so we can analyze which time horizon the agent is most predictive at. `--score-all` automatically scores all evaluations that have enough elapsed time.

### Pipeline sanity checks (Phase 1, 12 checks per ticker)

These verify the pipeline produced valid output. NOT evaluation of analytical quality.

| Check | What it verifies |
|-------|-----------------|
| `data_fetched` | yfinance returned data |
| `has_current_price`, `has_market_cap_bn`, `has_revenue_bn_ttm` | Key fields populated |
| `implied_cagr_computed` | Reverse DCF ran |
| `implied_cagr_sane` | CAGR between -30% and +80% |
| `narrative_count` | Exactly 3 narratives |
| `probability_sum` | Bull + base + bear ≈ 1.0 |
| `narrative_labels` | Labels are bull, base, bear |
| `gap_populated`, `gap_has_closest` | Expectations gap analysis completed |
| `brief_generated` | Final brief exists |

### Output structure

```
data/tickers.csv                     # Input basket (goes to GitHub)
evaluations/YYYY-MM-DD/              # Raw daily output (gitignored)
├── analysis_summary.csv             #   One row per ticker
├── sanity_summary.csv               #   Pipeline check results
├── {TICKER}_brief.txt               #   Full research brief
├── {TICKER}_state.json              #   Complete agent state
├── {TICKER}_sanity.json             #   Per-check details
└── scores_{N}d.csv                  #   Scoring results at N days
results/scoring_log.csv              # Append-only scored results (goes to GitHub)
```

### Ticker basket

30 tickers in `data/tickers.csv`, diversified across:
- **Sectors:** Technology, Healthcare, Financials, Energy, Industrials, Consumer, Utilities
- **Sizes:** Mega-cap ($500B+), large-cap ($50-500B), mid-cap ($10-50B)
- **Styles:** Growth, value, defensive, volatile
- **International ADRs:** BABA, TSM

---

## 10. Design Decisions & Trade-offs

### Why no query classification?

The LLM reads the user's query directly in the prompt. It adapts its analysis emphasis naturally. This is more robust than keyword matching (which can't handle ambiguous queries) and more agentic (the reasoning happens in the LLM, not in a dispatch table).

### Why hypothesis-first with business context only?

Most financial AI tools follow a "gather everything, then summarize" pattern. This leads to anchoring — if the LLM sees a P/E of 45x before forming narratives, every narrative will be shaped by that number. By giving it only qualitative context, narratives are constructed from business understanding. Financial data then tests narratives rather than anchoring them.

### Why build the brief in Python instead of an LLM call?

The brief structure is fixed. Sections like "Competing Narratives" and "Base Rate Check" are reformatted versions of data already in the state. Using an LLM to interpolate template variables is wasteful. The LLM is only needed for executive summary and confidence assessment.

### Why a simplified reverse DCF instead of just P/E comparison?

Mauboussin's framework is about reverse-engineering the growth rate embedded in the price. A P/E comparison is directional but imprecise. The simplified reverse DCF produces a falsifiable number ("the market prices in 25% revenue CAGR for 5 years") with stated assumptions.

### Why stubs instead of mocks?

Stubs return minimal but structurally valid data. The analyst always receives all expected state keys. Each stub documents the shape of data the real implementation must provide. Replacing a stub with a real agent is a drop-in swap.

### Why a fixed pipeline for POC instead of analyst-with-tools?

The ReAct-style analyst (Phase 2) is architecturally better but harder to debug, more expensive per run (multiple LLM tool-calling rounds), and harder to demonstrate in LangSmith traces. The fixed pipeline is predictable, cheap, and easy to demo. The migration path is clear: collapse `data_gathering + analyst` into a single agentic node, convert data functions into `@tool`-decorated functions.

---

## 11. Configuration

The LLM provider is configurable via environment variables in `.env`:

| Provider | Reasoning Model | Data Model | Cost |
|----------|----------------|------------|------|
| Groq | Llama 3.3 70B | Llama 3.3 70B | Free |
| Anthropic | Claude Sonnet 4.6 | Claude Haiku 4.5 | ~$0.03-0.10/query |

---

## 12. Phase 2+ Roadmap

| Phase | Feature | Impact |
|-------|---------|--------|
| **2a** | Analyst-with-tools (ReAct pattern) | Genuinely agentic: analyst calls tools during reasoning |
| **2b** | `fetch_market_context` tool (yfinance, no new API) | Systematic vs. idiosyncratic decomposition; multi-timeframe relative performance + VIX |
| **2c** | `search_news` tool (Tavily/Serper) | Ticker-specific AND macro news; enables contradiction detection |
| **2d** | `lookup_base_rates` tool (Damodaran datasets) | Empirical base rates replace hardcoded heuristics |
| **2e** | Evaluation framework (Strategy A + B) | Automated accuracy checks + earnings event scoring |
| **3a** | SEC filings tool (EDGAR + ChromaDB RAG) | Risk factors, insider transactions from primary sources |
| **3b** | Cross-run caching | Disk cache with TTLs per data type |
| **3c** | Multi-turn conversation | Follow-up queries that extend previous analysis |
| **3d** | LangSmith tracing | Visual reasoning traces for demos and debugging |
| **4** | Point-in-time backtesting | Gold-standard evaluation with paid historical data |
