"""
Judge — Final Synthesizer Agent.

Uses the ResourcePool to acquire a high-priority slot (criticality=10)
for the synthesis task, with full retry support.
"""

from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from core.resource_pool import ResourcePool

console = Console()


def evaluate_and_synthesize(
    clients: dict[int, genai.Client],
    pool: ResourcePool,
    query: str,
    final_context: str,
) -> tuple[str, str]:
    """
    Synthesize all agent outputs into a final polished answer.
    """
    # ── RAG: Retrieve evaluation guidelines + factual grounding ───────
    from core.rag_engine import get_judge_rag, format_rag_context
    rag_docs = get_judge_rag().retrieve(query, top_k=3)
    rag_section = format_rag_context(rag_docs)

    console.print(f"[dim]  📚 Judge RAG: retrieved {len(rag_docs)} docs "
                  f"(hybrid BM25+Vector)[/dim]")

    rag_note = ""
    if rag_docs:
        rag_note = f"""
    === EVALUATION GUIDELINES & REFERENCE KNOWLEDGE ===
    {rag_section}
    === END REFERENCE KNOWLEDGE ===
    """

    prompt = f"""
    You are the Final Synthesizer and Judge Agent. 
    The user originally asked this query:
    '{query}'
    {rag_note}
    A team of specialized AI agents have worked on this and generated the following combined work:
    --- CONTEXT START ---
    {final_context}
    --- CONTEXT END ---
    
    Your task is to review all the information provided by the agents, synthesize it, resolve any contradictions, and compile the final, polished answer to the user's query.
    If evaluation guidelines were provided above, use them to assess quality and structure your output.
    Format your final output using Markdown for better readability.
    """

    max_retries = 6
    last_error = None

    with console.status("[bold magenta]Judge Agent is evaluating and synthesizing...[/bold magenta]"):
        for attempt in range(1, max_retries + 1):
            slot = pool.acquire("Judge_Synthesizer", criticality=10)
            if slot is None:
                console.print(f"[bold yellow]⚠ Judge: could not acquire slot "
                              f"(attempt {attempt}/{max_retries})[/bold yellow]")
                import time
                time.sleep(2)
                continue

            try:
                console.print(
                    f"[dim]Judge using KEY{slot.key_index}/{slot.model} "
                    f"(attempt {attempt}/{max_retries})[/dim]"
                )

                config = types.GenerateContentConfig(temperature=0.7)
                response = slot.client.models.generate_content(
                    model=slot.model,
                    contents=prompt,
                    config=config,
                )

                if not response.text or not response.text.strip():
                    raise ValueError("Empty response from model")

                console.print("\n[bold magenta]=== FINAL RESULT ===[/bold magenta]")
                console.print(Markdown(response.text))
                return response.text, slot.model

            except Exception as e:
                last_error = e
                console.print(
                    f"[bold yellow]⚠ Judge failed on KEY{slot.key_index}/{slot.model}: "
                    f"{e} (attempt {attempt}/{max_retries})[/bold yellow]"
                )
                pool.blacklist_slot(slot)

            finally:
                pool.release(slot, "Judge_Synthesizer")

    console.print(f"[bold red]✘ Judge FAILED after {max_retries} attempts. Last error: {last_error}[/bold red]")
    return "", ""
