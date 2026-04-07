"""
Multi-Agent CLI — Entry Point

Wires together:
  - 4 API keys via get_clients()
  - ResourcePool with Semaphore + HashMap
  - Planner → Executor → Judge pipeline
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console

from llm import get_clients
from resource_pool import ResourcePool
from planner import build_plan
from executor import execute_plan
from judge import evaluate_and_synthesize


def main():
    load_dotenv()
    console = Console()

    # ── Load all API key clients ──────────────────────────────────────────
    try:
        clients = get_clients()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        console.print("Set GEMINI_API_KEY1 through GEMINI_API_KEY4 in the .env file.")
        sys.exit(1)

    # ── Create the shared ResourcePool ────────────────────────────────────
    pool = ResourcePool(clients)

    # ── Initialize RAG Engine (BM25 + Vector + Hybrid) ────────────────────
    from rag_engine import initialize_rag
    initialize_rag()

    console.print("\n[bold green]Welcome to the Gemini Multi-Agent CLI![/bold green]")
    console.print(f"[dim]Resource Pool: {len(clients)} API keys × 4 models = {len(pool._slots)} slots[/dim]")
    console.print("[dim]RAG Engine: BM25 (Planner) + Vector (Executor) + Hybrid (Judge)[/dim]")
    console.print("Type 'exit' or 'quit' to terminate.\n")

    while True:
        try:
            query = console.input("[bold yellow]Enter your complex task or query:[/bold yellow] ")
            if query.strip().lower() in ["exit", "quit"]:
                console.print("Goodbye!")
                break

            if not query.strip():
                continue

            # Phase 1: Planning
            with console.status("[bold cyan]Planner Agent is designing the solution...[/bold cyan]"):
                try:
                    plan = build_plan(clients, query)
                except Exception as e:
                    console.print(f"[bold red]Planning Error:[/bold red] {e}")
                    continue

            console.print("\n[bold]Generated Execution Plan:[/bold]")
            for i, agent in enumerate(plan.agents):
                tools_str = (
                    f" (Tools: {', '.join(agent.tools_required)})"
                    if agent.tools_required
                    else ""
                )
                console.print(
                    f"  {i+1}. [bold cyan]{agent.name}[/bold cyan] "
                    f"[dim](crit={agent.criticality})[/dim]{tools_str}: {agent.task}"
                )

            # Phase 2: Execution
            final_context = execute_plan(clients, plan, pool)
            if not final_context.strip():
                console.print("[bold red]Agents failed to produce any context.[/bold red]")
                continue

            # Phase 3: Synthesize
            evaluate_and_synthesize(clients, pool, query, final_context)

            console.print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            console.print("\nOperation cancelled by user. Type 'exit' to quit.")
        except Exception as e:
            console.print(f"\n[bold red]Unexpected Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
