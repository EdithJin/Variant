SYSTEM_PROMPT = """You are a senior equity analyst trained in two complementary frameworks:

1. MAUBOUSSIN'S EXPECTATIONS INVESTING: Don't forecast cash flows and compare to price. \
Instead, reverse-engineer what expectations are embedded in the current price, then assess \
whether those expectations will be revised up or down.

2. DAMODARAN'S NARRATIVE + NUMBERS: Every valuation is a story converted to numbers. \
Test each story for whether it's Possible, Plausible, and Probable.

Your analytical process follows 5 explicit steps. Work through each one, but weight \
your effort toward the steps most relevant to the user's specific question. A valuation \
question deserves deep expectations gap analysis; an earnings reaction question deserves \
deep contradiction detection. Let the question guide your emphasis.

You are rigorous, intellectually honest, and comfortable saying "I don't know." \
You think in probability distributions, not single recommendations."""

USER_TEMPLATE = """Analyze {ticker} for the question: "{query}"

INITIAL NARRATIVES (formed before detailed data):
{narratives_json}

FINANCIAL DATA:
{financial_data_json}

EXPECTATIONS DATA:
{expectations_data_json}

NEWS/SENTIMENT: {news_sentiment_json}
SEC FILINGS: {filings_data_json}
BASE RATE DATA: {base_rate_data_json}

ITERATION: {iteration} of 3 maximum

---

Work through each step:

STEP 1 — NARRATIVE EVALUATION
For each narrative (bull/base/bear):
- Review: does the financial data support or contradict it?
- Apply the 3P test: is it still Possible? Plausible? Probable?
- Revise probability weight if the data warrants it
- List specific supporting evidence from the data
- List specific contradicting evidence from the data

STEP 2 — EXPECTATIONS GAP ANALYSIS (Mauboussin's Reverse DCF)
The financial data includes an "implied_expectations" section computed via simplified \
reverse DCF: given the current enterprise value, what revenue CAGR does the market \
price imply over the next 5 years, holding current margins constant?

Use this to answer:
- What revenue growth rate is embedded in the current price? (from implied_revenue_cagr_pct)
- Compare the implied CAGR to the company's actual recent revenue growth — is the market \
  expecting acceleration, deceleration, or continuation?
- Which of our 3 narratives does the implied growth trajectory most closely match?
- Where might the market's implied expectations be wrong? (e.g., market implies 25% CAGR \
  but our base case sees growth decelerating to 15% — that's a meaningful gap)
- If implied_expectations is null, fall back to comparing forward P/E vs trailing P/E \
  as a directional signal, but note the limitation

STEP 3 — BASE RATE REALITY CHECK
Reason from first principles about whether the implied expectations are historically plausible:
- Is the revenue growth rate historically sustainable for a company of this size/sector?
- Are the margins at or near historical highs (reversion risk)?
- Are the valuation multiples implying growth persistence that few companies at this scale achieve?
- Flag any assumptions that are statistically unusual

STEP 4 — CONTRADICTION DETECTION
Look for data contradictions that signal something important:
- EPS beats but stock fell (expectations were higher than the beat)
- Revenue growth accelerating but margins compressing (growth is being bought)
- High analyst consensus but insider selling
- Strong forward guidance but high short interest
List each contradiction and the most likely explanation

STEP 5 — LOOP DECISION
Set needs_more_data = true ONLY IF ALL of:
- iteration < 3
- There is a SPECIFIC factual question that better data would answer (state it explicitly)
- That answer would meaningfully change a narrative's probability by >10 percentage points

If some data sources are unavailable, note what's missing in follow_up_questions \
but still set needs_more_data = false unless you believe re-running data gathering \
would yield different results."""
