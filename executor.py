"""
Executor — dispatches agent tasks through the ResourcePool.

Agents are sorted by criticality (highest first) and submitted to a
ThreadPoolExecutor. The ResourcePool handles model selection, semaphore-
based mutual exclusion, HashMap logging, and retry logic.
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
