"""
Planner — uses structured JSON schema output to break a user query
into independent sub-tasks for concurrent agent execution.
"""

import json
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from rich.console import Console

console = Console()

PLANNER_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
]


class AgentConfig(BaseModel):
    name: str = Field(description="Name of the agent, e.g., 'Researcher', 'Data_Analyst'")
    role: str = Field(description="The system prompt/role describing how this agent should behave")
    task: str = Field(description="The specific task this agent needs to execute")
    tools_required: List[str] = Field(description="List of tools the agent might need, e.g., ['google_search']")
    criticality: int = Field(
        description="Priority / criticality of this task (1-10, 10 being most complex/critical)",
        default=5,
    )


class AgentPlan(BaseModel):
    agents: List[AgentConfig] = Field(
        description="List of independent agents to execute concurrently to solve the user query"
    )


def build_plan(clients: dict[int, genai.Client], query: str) -> AgentPlan:
    """
    Break a user query into an AgentPlan using structured output.
    """
    # ── RAG: Retrieve relevant planning context via BM25 ──────────────
    rag_docs = []
    rag_section = ""
    try:
        from core.rag_engine import get_planner_rag, format_rag_context
        rag_docs = get_planner_rag().retrieve(query, top_k=3)
        rag_section = format_rag_context(rag_docs)
        console.print(f"[dim]  📚 Planner RAG: retrieved {len(rag_docs)} relevant planning docs[/dim]")
    except Exception as exc:
        console.print(
            f"[dim]  📚 Planner RAG unavailable; continuing without planning knowledge base ({exc})[/dim]"
        )
    rag_note = f"\n    === REFERENCE KNOWLEDGE (from internal planning knowledge base) ===\n    {rag_section}\n    === END REFERENCE KNOWLEDGE ===\n" if rag_docs else ""
    prompt = f"""
    You are an expert AI Planner. The user has provided the following query:
    '{query}'
    {rag_note}
    Your task is to break down this complex query into independent sub-tasks that AI agents can solve simultaneously.
    Define a list of agents. For each agent, provide a name, a detailed role, the specific task it must accomplish, the tools it might need, and assign a criticality score from 1-10 indicating how complex or important the specific task is.
    The agents will be executed CONCURRENTLY. They cannot depend on each other's outputs. You must make their tasks completely independent.
    Keep the plan as concise as possible but ensure it comprehensively addresses the user's query.
    If the task requires searching the internet, use the tool 'google_search'.
    Use the reference knowledge above (if provided) to inform your planning decisions, agent roles, criticality scoring, and decomposition strategy.
    """

    json_prompt = prompt + """
    
    IMPORTANT: You MUST respond with ONLY valid JSON in this exact schema, nothing else:
    {
        "agents": [
            {
                "name": "agent_name",
                "role": "agent role description",
                "task": "specific task",
                "tools_required": ["google_search"],
                "criticality": 7
            }
        ]
    }
    """

    last_error = None
    gemma_models = {"gemma-4-31b-it", "gemma-4-26b-a4b-it"}

    for model in PLANNER_MODELS:
        for key_idx, client in clients.items():
            try:
                console.print(f"[dim]Planner trying KEY{key_idx}/{model}...[/dim]")

                is_gemma = model in gemma_models

                if is_gemma:
                    response = client.models.generate_content(
                        model=model,
                        contents=json_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                        ),
                    )
                else:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=AgentPlan,
                        ),
                    )

                raw_text = response.text.strip()

                if is_gemma and raw_text.startswith("```"):
                    lines = raw_text.split("\n")
                    json_lines = []
                    in_block = False
                    for line in lines:
                        if line.strip().startswith("```") and not in_block:
                            in_block = True
                            continue
                        elif line.strip() == "```" and in_block:
                            break
                        elif in_block:
                            json_lines.append(line)
                    raw_text = "\n".join(json_lines)

                data = json.loads(raw_text)
                plan = AgentPlan(**data)
                console.print(f"[bold green]✔ Planner succeeded on KEY{key_idx}/{model}[/bold green]")
                return plan

            except Exception as e:
                error_str = str(e).lower()
                last_error = e

                if any(k in error_str for k in ["429", "resource_exhausted", "quota"]):
                    console.print(
                        f"[bold yellow]⚠ Planner: KEY{key_idx}/{model} "
                        f"quota exhausted, trying next...[/bold yellow]"
                    )
                elif "json" in error_str or "parse" in error_str or "decode" in error_str:
                    console.print(
                        f"[bold yellow]⚠ Planner: KEY{key_idx}/{model} "
                        f"bad JSON response, trying next...[/bold yellow]"
                    )
                else:
                    console.print(
                        f"[bold yellow]⚠ Planner: KEY{key_idx}/{model} "
                        f"error: {e}, trying next...[/bold yellow]"
                    )
                continue

    raise RuntimeError(f"All planner (key, model) combinations exhausted. Last error: {last_error}")
