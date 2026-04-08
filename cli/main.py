"""
Multi-Agent CLI — Entry Point

Wires together:
  - 4 API keys via get_clients()
  - ResourcePool with Semaphore + HashMap
  - Planner → Executor → Judge pipeline
  - Persistent Storage (SQLite + ChromaDB)
  - Multi-Level Query Cache (L1 hash + L2 semantic)
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from core.llm import get_clients
from core.resource_pool import ResourcePool
from core.planner import build_plan
from core.executor import execute_plan
from core.judge import evaluate_and_synthesize


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
    from core.rag_engine import initialize_rag
    initialize_rag()

    # ── Initialize Persistent Storage (SQLite + ChromaDB) ─────────────────
    from core.storage import StorageManager
    storage = StorageManager()

    # ── Initialize Query Cache (L1 hash + L2 semantic) ────────────────────
    from core.cache import QueryCache
    cache = QueryCache()

    console.print("\n[bold green]Welcome to the Gemini Multi-Agent CLI![/bold green]")
    console.print(f"[dim]Resource Pool: {len(clients)} API keys × 4 models = {len(pool._slots)} slots[/dim]")
    console.print("[dim]RAG Engine: BM25 (Planner) + Vector (Executor) + Hybrid (Judge)[/dim]")
    console.print(f"[dim]Storage: SQLite + ChromaDB (user KB: {storage.get_user_kb_count()} docs)[/dim]")
    cache_stats = cache.stats()
    console.print(f"[dim]Cache: L1 ({cache_stats['l1_entries']} entries) + L2 ({cache_stats['l2_entries']} entries)[/dim]")
    console.print("\nCommands: '/exit' or '/quit' | '/history' | '/clearcache' | '/cachestats'\n")

    while True:
        try:
            query = console.input("[bold yellow]Enter your complex task or query:[/bold yellow] ")
            stripped = query.strip().lower()

            if stripped in ["/exit", "/quit"]:
                console.print("Goodbye!")
                break

            if not query.strip():
                continue

            if stripped == "/history":
                storage.print_history(limit=10)
                continue

            if stripped == "/clearcache":
                cache.clear()
                continue

            if stripped == "/cachestats":
                cache.print_stats()
                continue

            # ── Cache Lookup ──────────────────────────────────────────
            cached = cache.lookup(query.strip())
            if cached is not None:
                console.print(
                    f"\n[bold green]⚡ CACHE HIT ({cached.cache_level}, "
                    f"similarity: {cached.similarity:.3f})[/bold green]"
                )
                console.print(f"[dim]Original query: {cached.query_text[:80]}...[/dim]")
                console.print(f"[dim]Hit count: {cached.hit_count}[/dim]\n")
                console.print("[bold magenta]=== CACHED RESULT ===[/bold magenta]")
                console.print(Markdown(cached.cached_answer))
                console.print("\n" + "=" * 60 + "\n")
                continue

            console.print("[dim]🔍 Cache MISS — running full pipeline...[/dim]")

            t_start = time.time()

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
                    f"[dim](crit={agent.criticality})[/dim]{tools_str}: {agent.task[:100]}"
                )

            # Phase 2: Execution
            final_context = execute_plan(clients, plan, pool)
            if not final_context.strip():
                console.print("[bold red]Agents failed to produce any context.[/bold red]")
                continue

            # Phase 3: Synthesize
            final_answer, model_used = evaluate_and_synthesize(
                clients, pool, query, final_context
            )

            duration = time.time() - t_start

            if final_answer.strip():
                plan_json = json.dumps(
                    {"agents": [a.model_dump() for a in plan.agents]},
                    indent=2,
                )
                storage.save_conversation(
                    query=query.strip(),
                    plan_json=plan_json,
                    agent_outputs=[],
                    final_answer=final_answer,
                    model_used=model_used,
                    duration_seconds=duration,
                )
                cache.store(query.strip(), final_answer)

                from core.rag_engine import add_user_interaction
                add_user_interaction(query.strip(), final_answer)

                console.print(
                    f"\n[dim]⏱ Total pipeline time: {duration:.1f}s | "
                    f"Answer saved to storage + cache[/dim]"
                )

            console.print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            console.print("\nOperation cancelled by user. Type 'exit' to quit.")
        except Exception as e:
            console.print(f"\n[bold red]Unexpected Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
