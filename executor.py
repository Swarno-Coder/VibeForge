"""
Executor — dispatches agent tasks through the ResourcePool.

Agents are sorted by criticality (highest first) and submitted to a
ThreadPoolExecutor. The ResourcePool handles model selection, semaphore-
based mutual exclusion, HashMap logging, and retry logic.

RAG Integration: Before dispatching, each agent's task is enriched with
semantically relevant knowledge from the Vector RAG (ChromaDB + sentence-transformers).
This happens BEFORE pool interaction — the ResourcePool is NOT modified.
"""

from google import genai
from planner import AgentPlan
from resource_pool import ResourcePool
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console

console = Console()


def execute_plan(clients: dict[int, genai.Client], plan: AgentPlan, pool: ResourcePool) -> str:
    """
    Execute all agents in the plan concurrently via the ResourcePool.

    RAG Integration: Each agent's task text is enriched with semantically
    relevant documents from the Vector retriever before being submitted to
    the pool's execute_with_retry(). This adds domain-specific grounding
    without touching any pool logic.

    Args:
        clients: dict of {key_index: genai.Client}
        plan:    the AgentPlan from the planner
        pool:    shared ResourcePool instance
    """
    accumulated_context = ""

    # Sort agents by criticality (highest first) → priority queue
    sorted_agents = sorted(
        plan.agents,
        key=lambda a: getattr(a, "criticality", 5),
        reverse=True,
    )

    # ── RAG: Enrich each agent's task with Vector-retrieved knowledge ──
    from rag_engine import get_executor_rag, format_rag_context
    retriever = get_executor_rag()

    console.print("\n[bold blue]═══ Enriching Agents with RAG Context ═══[/bold blue]")
    for agent in sorted_agents:
        rag_docs = retriever.retrieve(agent.task, top_k=3)
        if rag_docs:
            rag_ctx = format_rag_context(rag_docs)
            agent.task = (
                agent.task
                + "\n\n=== RELEVANT KNOWLEDGE (from internal knowledge base — use to ground your analysis) ===\n"
                + rag_ctx
                + "\n=== END RELEVANT KNOWLEDGE ==="
            )
            console.print(
                f"[dim]  📚 {agent.name}: enriched with {len(rag_docs)} RAG docs "
                f"(top score: {rag_docs[0].score:.3f})[/dim]"
            )
        else:
            console.print(f"[dim]  📚 {agent.name}: no relevant RAG docs found[/dim]")

    console.print("\n[bold blue]═══ Starting Priority-Queued Concurrent Agent Execution ═══[/bold blue]")
    console.print(f"[dim]Agents queued: {len(sorted_agents)} | "
                  f"Resource slots: {len(pool._slots)} | "
                  f"Max retries per agent: 12 (with tier demotion)[/dim]\n")

    for agent in sorted_agents:
        crit = getattr(agent, "criticality", 5)
        console.print(f"  [cyan]⏳ Queued:[/cyan] {agent.name} (criticality={crit})")

    console.print()

    # Submit all agents concurrently — pool's semaphore limits actual concurrency
    futures = []
    with ThreadPoolExecutor(max_workers=len(pool._slots)) as executor:
        for agent in sorted_agents:
            futures.append(
                executor.submit(pool.execute_with_retry, agent, 12)
            )

        with console.status("[bold cyan]Agents executing across resource pool...[/bold cyan]"):
            for future in as_completed(futures):
                accumulated_context += future.result()
                # Print live allocation snapshot after each completion
                pool.print_allocation_table()

    console.print("\n[bold blue]═══ All Agents Finished ═══[/bold blue]")

    # Save full allocation log to file
    pool.save_log_to_file()

    return accumulated_context

