"""
BBL AI Engineer Programming Test — CLI Entrypoint.

Provides an interactive and batch interface for querying the multi-agent
retrieval pipeline.  Run directly with:

    python main.py              # interactive mode
    python main.py --demo       # run pre-defined demo queries
    python main.py -q "..."     # single one-shot query

Uses ``rich`` for polished terminal output.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from src.workflow import run_pipeline

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rich console
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# Demo queries
# ---------------------------------------------------------------------------
DEMO_QUERIES: list[str] = [
    "What is the policy on international travel?",
    "Can I work remotely from another country?",
    "What are the rules regarding annual leave carry-over?",
    "What is the policy on buying a company vehicle?",  # Edge case: NO_DATA_FOUND
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def display_result(query: str, result: dict, index: int | None = None) -> None:
    """Pretty-print a single pipeline result with Rich panels.

    Args:
        query: The user's original question.
        result: The dict returned by ``run_pipeline``.
        index: Optional query number for batch display.
    """
    # ── Header ──────────────────────────────────────────────────────────
    label = f"Query {index}" if index is not None else "Query"
    console.print()
    console.print(Rule(f"[bold cyan]{label}[/bold cyan]", style="cyan"))

    # ── User Query ──────────────────────────────────────────────────────
    console.print(Panel(
        Text(query, style="bold white"),
        title="[bold blue]User Query[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))

    # ── Retrieved Snippets ──────────────────────────────────────────────
    snippets = result.get("retrieved_snippets", "")
    if snippets == "NO_RELEVANT_DATA_FOUND":
        snippet_display = Text("NO_RELEVANT_DATA_FOUND", style="bold red")
    else:
        snippet_display = Text(snippets, style="dim white")

    console.print(Panel(
        snippet_display,
        title="[bold yellow]Data Retriever Output (Raw Snippets)[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    ))

    # ── Final Report ────────────────────────────────────────────────────
    final_report = result.get("final_report", "")
    console.print(Panel(
        Markdown(final_report),
        title="[bold green]Report Generator Output (Synthesized Answer)[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # ── Execution Logs ──────────────────────────────────────────────────
    logs = result.get("logs", [])
    if logs:
        log_text = "\n".join(logs)
        console.print(Panel(
            Text(log_text, style="dim cyan"),
            title="[bold dim]Pipeline Logs[/bold dim]",
            border_style="dim",
            padding=(0, 2),
        ))


def print_banner() -> None:
    """Print the application banner."""
    banner = """
[bold cyan]  ____  ____  _         _    ___   _____           _   
 | __ )| __ )| |       / \\  |_ _| |_   _|___  ___| |_ 
 |  _ \\|  _ \\| |      / _ \\  | |    | | / _ \\/ __| __|
 | |_) | |_) | |___  / ___ \\ | |    | ||  __/\\__ \\ |_ 
 |____/|____/|_____|/_/   \\_\\___|   |_| \\___||___/\\__|[/bold cyan]
                                                        
[dim]Multi-Agent Knowledge Retrieval Pipeline[/dim]
[dim]LangChain + LangGraph  |  Two-Agent Architecture[/dim]
"""
    console.print(banner)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_demo() -> None:
    """Run all pre-defined demo queries and display results."""
    print_banner()
    console.print(
        f"[bold magenta]Running {len(DEMO_QUERIES)} demo queries...[/bold magenta]\n"
    )

    for i, query in enumerate(DEMO_QUERIES, start=1):
        console.print(
            f"[bold cyan][{i}/{len(DEMO_QUERIES)}][/bold cyan] Processing: "
            f"[italic]{query}[/italic]"
        )
        start = time.time()
        result = run_pipeline(query)
        elapsed = time.time() - start

        display_result(query, result, index=i)
        console.print(f"[dim]Completed in {elapsed:.2f}s[/dim]\n")

    console.print(Rule("[bold green]All demo queries completed[/bold green]", style="green"))


def run_single(query: str) -> None:
    """Run a single one-shot query."""
    print_banner()
    console.print(f"[bold magenta]Processing query...[/bold magenta]\n")

    start = time.time()
    result = run_pipeline(query)
    elapsed = time.time() - start

    display_result(query, result)
    console.print(f"\n[dim]Completed in {elapsed:.2f}s[/dim]")


def run_interactive() -> None:
    """Run an interactive query loop (type 'exit' or 'quit' to stop)."""
    print_banner()
    console.print(
        "[bold magenta]Interactive mode[/bold magenta] — "
        "type your question and press Enter.\n"
        "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.\n"
    )

    query_num = 0
    while True:
        try:
            console.print("[bold blue]>[/bold blue] ", end="")
            query = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        query_num += 1
        start = time.time()
        result = run_pipeline(query)
        elapsed = time.time() - start

        display_result(query, result, index=query_num)
        console.print(f"[dim]Completed in {elapsed:.2f}s[/dim]\n")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="BBL AI Pipeline",
        description="Multi-Agent Knowledge Retrieval Pipeline for the BBL AI Engineer Test.",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Run a single query and exit.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run pre-defined demo queries automatically.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""
    args = parse_args()

    if args.demo:
        run_demo()
    elif args.query:
        run_single(args.query)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
