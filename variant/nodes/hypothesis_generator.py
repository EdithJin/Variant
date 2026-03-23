"""
Hypothesis generator node.

This is the first LLM-powered step. It constructs 3 competing narratives
(bull/base/bear) BEFORE detailed data analysis. This enforces Damodaran's
hypothesis-first discipline: form your stories about the business, then
gather data to test them — not the other way around.

The node receives ONLY qualitative business context (company name, sector,
industry, description) — no financial ratios or multiples. This prevents
anchoring bias: narratives are constructed from business understanding,
and financial numbers are gathered AFTER to test them.
"""
from typing import Optional

from pydantic import BaseModel, Field

from variant.state import AgentState, Narrative
from variant.prompts.hypothesis_generator import SYSTEM_PROMPT, USER_TEMPLATE
from variant.config import get_llm


# ── Pydantic schemas for structured LLM output ────────────────────────
# These ensure the LLM returns valid JSON matching our Narrative type.

class NarrativeModel(BaseModel):
    label: str = Field(description="'bull', 'base', or 'bear'")
    story: str = Field(description="2-3 sentence narrative about the company's future")
    probability: float = Field(description="Probability weight 0.0-1.0, all three must sum to 1.0")
    key_assumptions: list[str] = Field(description="2-3 key assumptions that must hold")
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)


class NarrativesOutput(BaseModel):
    narratives: list[NarrativeModel] = Field(description="Exactly 3 narratives: bull, base, bear")


def _format_business_context(business_context: Optional[dict]) -> str:
    """
    Formats ONLY qualitative business context — no financial ratios or multiples.
    """
    if not business_context:
        return "No business context available. Construct narratives from your general knowledge of this company."
    if business_context.get("error"):
        return f"Data fetch error: {business_context['error']}. Construct narratives from general knowledge of this company."
    bc = business_context
    lines = []
    if bc.get("company_name"):
        lines.append(f"Company: {bc['company_name']}")
    if bc.get("sector") or bc.get("industry"):
        lines.append(f"Sector / Industry: {bc.get('sector', 'N/A')} / {bc.get('industry', 'N/A')}")
    if bc.get("country"):
        lines.append(f"Headquarters: {bc['country']}")
    if bc.get("full_time_employees"):
        lines.append(f"Employees: {bc['full_time_employees']:,}")
    if bc.get("business_summary"):
        # Truncate to ~500 chars to keep prompt focused
        summary = bc["business_summary"]
        if len(summary) > 500:
            summary = summary[:497] + "..."
        lines.append(f"\nBusiness description:\n{summary}")
    return "\n".join(lines) if lines else "No business context available."


def hypothesis_generator_node(state: AgentState) -> dict:
    llm = get_llm(structured_output_schema=NarrativesOutput)

    business_context = _format_business_context(state.get("business_context"))
    user_message = USER_TEMPLATE.format(
        ticker=state["ticker"],
        query=state["query"],
        business_context=business_context,
    )

    result: NarrativesOutput = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    narratives: list[Narrative] = [
        {
            "label": n.label,
            "story": n.story,
            "probability": n.probability,
            "key_assumptions": n.key_assumptions,
            "supporting_evidence": n.supporting_evidence,
            "contradicting_evidence": n.contradicting_evidence,
        }
        for n in result.narratives
    ]

    return {"narratives": narratives}
