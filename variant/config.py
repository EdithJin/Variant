"""
LLM provider configuration.

Supports two providers:
  - anthropic (default) — Claude models, requires ANTHROPIC_API_KEY
  - groq (free tier)    — Llama models via Groq, requires GROQ_API_KEY

All LLM access in the project goes through get_llm(). This keeps provider
switching centralized — changing LLM_PROVIDER in .env switches every node.

Two model tiers:
  - REASONING_MODEL: used by hypothesis_generator, analyst, synthesis
    (needs strong structured output and nuanced reasoning)
  - DATA_MODEL: reserved for future data extraction nodes
    (can be cheaper/faster since it just formats data)
"""
import os

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()

_DEFAULTS = {
    "anthropic": {
        "reasoning": "claude-sonnet-4-6",
        "data": "claude-haiku-4-5-20251001",
    },
    "groq": {
        "reasoning": "llama-3.3-70b-versatile",
        "data": "llama-3.3-70b-versatile",
    },
}

REASONING_MODEL = os.environ.get("REASONING_MODEL", _DEFAULTS[LLM_PROVIDER]["reasoning"])
DATA_MODEL = os.environ.get("DATA_MODEL", _DEFAULTS[LLM_PROVIDER]["data"])


def get_llm(model: str = None, structured_output_schema=None):
    """
    Factory that returns a LangChain chat model for the configured provider.

    Args:
        model: Override the model ID. Defaults to REASONING_MODEL.
        structured_output_schema: A Pydantic model class. If provided, wraps
            the LLM with .with_structured_output() so it returns parsed
            Pydantic objects instead of raw text.

    Returns:
        A LangChain chat model (ChatGroq or ChatAnthropic), optionally
        wrapped for structured output.
    """
    m = model or REASONING_MODEL

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=m, api_key=os.environ["GROQ_API_KEY"])
    else:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=m, api_key=os.environ["ANTHROPIC_API_KEY"])

    if structured_output_schema:
        return llm.with_structured_output(structured_output_schema)
    return llm
