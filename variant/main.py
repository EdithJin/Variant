"""
Variant — Financial Research Agent
Usage: python -m variant.main TICKER "QUESTION"
Example: python -m variant.main NVDA "Is NVDA overvalued?"
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

# Load .env from project root (one level up from this file, or current dir)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

from variant.config import LLM_PROVIDER
from variant.graph import build_graph


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m variant.main TICKER 'QUESTION'")
        print("Example: python -m variant.main NVDA 'Is NVDA overvalued?'")
        sys.exit(1)

    required_key = "GROQ_API_KEY" if LLM_PROVIDER == "groq" else "ANTHROPIC_API_KEY"
    if not os.environ.get(required_key):
        print(f"Error: {required_key} not set. Check your .env file.")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    query = " ".join(sys.argv[2:])

    console = Console()
    console.print(Rule(f"[bold blue]Variant Research Agent[/bold blue]"))
    console.print(f"[dim]Ticker:[/dim] [bold]{ticker}[/bold]  |  [dim]Query:[/dim] {query}")
    console.print()

    initial_state = {
        "query": query,
        "ticker": ticker,
        "narratives": [],
        "business_context": None,
        "financial_data": None,
        "expectations_data": None,
        "news_sentiment": None,
        "filings_data": None,
        "base_rate_data": None,
        "expectations_gap": None,
        "contradictions": [],
        "base_rate_flags": [],
        "follow_up_questions": [],
        "iteration": 0,
        "needs_more_data": False,
        "analyst_reasoning_summary": None,
        "final_brief": None,
    }

    graph = build_graph()

    with console.status("[bold green]Analyzing...", spinner="dots"):
        brief = None
        seen = set()
        for event in graph.stream(initial_state, stream_mode="values"):
            if event.get("business_context") and "context" not in seen:
                seen.add("context")
                console.print(f"[dim]✓ Business context loaded ({event['business_context'].get('company_name', '')})[/dim]")
            if event.get("narratives") and not event.get("financial_data") and "narratives" not in seen:
                seen.add("narratives")
                console.print(f"[dim]✓ {len(event['narratives'])} narratives constructed (before financial data)[/dim]")
            if event.get("financial_data") and event.get("narratives") and "data" not in seen:
                seen.add("data")
                console.print("[dim]✓ Financial data fetched — testing narratives[/dim]")
            if event.get("expectations_gap"):
                iteration = event.get("iteration", 0)
                console.print(f"[dim]✓ Analysis complete (iteration {iteration})[/dim]")
            if event.get("final_brief"):
                brief = event["final_brief"]

    console.print()
    if brief:
        console.print(Panel(
            Text(brief),
            border_style="blue",
            padding=(1, 2),
        ))
    else:
        console.print("[red]No brief generated — check logs for errors.[/red]")


if __name__ == "__main__":
    main()
