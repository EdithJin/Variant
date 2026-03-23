"""
Synthesis node — formats the final research brief.

The brief structure is fixed, so most of it is built in Python (string
formatting). The LLM is called ONLY for the two parts that require
judgment: the executive summary and the confidence assessment.

This avoids using an expensive LLM call as a string formatter.
"""
from variant.state import AgentState
from variant.prompts.synthesis import SYSTEM_PROMPT, USER_TEMPLATE
from variant.config import get_llm


def _get_narrative(narratives: list[dict], label: str) -> dict:
    for n in narratives:
        if n.get("label", "").lower() == label.lower():
            return n
    return {}


def _format_narrative_block(n: dict) -> str:
    if not n:
        return "  [Not available]"
    pct = round(n.get("probability", 0) * 100)
    lines = [f"{n.get('label', '?').title()} Case ({pct}%): {n.get('story', 'N/A')}"]
    assumptions = n.get("key_assumptions", [])
    if assumptions:
        lines.append(f"  Key assumptions: {', '.join(assumptions)}")
    supporting = n.get("supporting_evidence", [])
    lines.append(f"  Supporting: {', '.join(supporting) if supporting else 'None identified'}")
    contradicting = n.get("contradicting_evidence", [])
    lines.append(f"  Against: {', '.join(contradicting) if contradicting else 'None identified'}")
    return "\n".join(lines)


def _build_brief_body(state: AgentState) -> str:
    """Build the fixed-structure portions of the brief in Python. No LLM needed."""
    narratives = state.get("narratives", [])
    bull = _get_narrative(narratives, "bull")
    base = _get_narrative(narratives, "base")
    bear = _get_narrative(narratives, "bear")

    gap = state.get("expectations_gap") or {}
    contradictions = state.get("contradictions", [])
    base_rate_flags = state.get("base_rate_flags", [])
    follow_up_questions = state.get("follow_up_questions", [])

    implied_growth = gap.get("price_implied_growth_pct")
    implied_str = f"{implied_growth}% annual revenue CAGR" if implied_growth is not None else "N/A"

    sections = []

    # Competing Narratives
    sections.append("COMPETING NARRATIVES")
    sections.append(_format_narrative_block(bull))
    sections.append("")
    sections.append(_format_narrative_block(base))
    sections.append("")
    sections.append(_format_narrative_block(bear))

    # Expectations Analysis
    sections.append("")
    sections.append("EXPECTATIONS ANALYSIS")
    sections.append(f"Price implies: {implied_str}")
    sections.append(f"Closest to our scenarios: {gap.get('closest_narrative', 'N/A')} case")
    sections.append(f"Assessment: {gap.get('gap_assessment', 'N/A')}")

    # Base Rate Check
    sections.append("")
    sections.append("BASE RATE CHECK")
    if base_rate_flags:
        for f in base_rate_flags:
            sections.append(f"- {f}")
    else:
        sections.append("No significant base rate anomalies flagged.")

    # Key Contradictions
    sections.append("")
    sections.append("KEY CONTRADICTIONS")
    if contradictions:
        for c in contradictions:
            sections.append(f"- {c}")
    else:
        sections.append("None identified.")

    # What We Don't Know
    sections.append("")
    sections.append("WHAT WE DON'T KNOW")
    if follow_up_questions:
        for q in follow_up_questions:
            sections.append(f"- {q}")
    else:
        sections.append("Analysis appears complete with available data.")

    return "\n".join(sections)


def _build_llm_context(state: AgentState) -> tuple:
    """Build a concise summary for the LLM to write exec summary + confidence."""
    narratives = state.get("narratives", [])
    lines = []
    for n in narratives:
        pct = round(n.get("probability", 0) * 100)
        lines.append(f"- {n.get('label', '?').title()} ({pct}%): {n.get('story', '')}")

    gap = state.get("expectations_gap") or {}
    implied = gap.get("price_implied_growth_pct")
    gap_line = f"Implied revenue CAGR: {implied}%. {gap.get('gap_assessment', '')}" if implied else gap.get("gap_assessment", "N/A")

    contradictions = state.get("contradictions", [])
    contra_str = "; ".join(contradictions) if contradictions else "None"

    return lines, gap_line, contra_str


def synthesis_node(state: AgentState) -> dict:
    # 1. Build the fixed-structure body in Python (no LLM)
    body = _build_brief_body(state)

    # 2. Call LLM ONLY for executive summary + confidence (requires judgment)
    narratives_lines, gap_line, contra_str = _build_llm_context(state)

    user_message = USER_TEMPLATE.format(
        ticker=state["ticker"],
        query=state["query"],
        analyst_reasoning_summary=state.get("analyst_reasoning_summary", ""),
        narratives_summary="\n".join(narratives_lines),
        expectations_gap_summary=gap_line,
        contradictions_summary=contra_str,
    )

    response = get_llm().invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    # Parse the two fields from the response
    response_text = response.content
    exec_summary = ""
    confidence = ""
    for line in response_text.strip().split("\n"):
        if line.startswith("EXECUTIVE_SUMMARY:"):
            exec_summary = line.replace("EXECUTIVE_SUMMARY:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            confidence = line.replace("CONFIDENCE:", "").strip()

    # If parsing fails, use the raw response as executive summary
    if not exec_summary:
        exec_summary = response_text.strip()

    # 3. Assemble the final brief
    divider = "=" * 55
    iteration = state.get("iteration", 1)

    brief = f"""{divider}
VARIANT RESEARCH BRIEF: {state["ticker"]}
Query: {state["query"]}
{divider}

EXECUTIVE SUMMARY
{exec_summary}

{body}

DATA & CONFIDENCE
Coverage: Financial data (yfinance) | Stubs: news, SEC filings, base rates
Iterations: {iteration}
Confidence: {confidence or 'Not assessed'}
{divider}"""

    return {"final_brief": brief}
