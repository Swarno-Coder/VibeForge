"""
Planner — uses structured JSON schema output to break a user query
into independent sub-tasks for concurrent agent execution.

Uses all 4 API keys × best models for fallback on quota exhaustion.
Model priority: Gemini 3.x → 2.5 → Gemma 4 (structured output support varies).
"""

import json
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from rich.console import Console

console = Console()

# Models that support structured JSON schema output, in priority order
# Gemini models support response_schema; Gemma models may not — we try and fall back
PLANNER_MODELS = [
    "gemini-3.1-flash-lite-preview",   # Best: thinking + reasoning
    "gemini-3-flash-preview",          # Very strong reasoning
    "gemini-2.5-flash",                # Solid general purpose
    "gemini-2.5-flash-lite",           # Good fallback
    "gemma-4-31b-it",                  # Gemma 4 — may support structured out
    "gemma-4-26b-a4b-it",             # Gemma 4 fallback
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

    Tries every (client, model) combination for resilience:
      - Iterates models in priority order (best → weakest)
      - For each model, tries all 4 API keys
      - Falls back on 429 / quota / JSON errors / any errors

    For Gemma models that don't support response_schema, falls back
    to plain text generation with JSON instruction in the prompt.

    Args:
        clients: {1: genai.Client, 2: genai.Client, ...}
        query:   the user's input query
    """
    prompt = f"""
    You are an expert AI Planner. The user has provided the following query:
    '{query}'
    
    Your task is to break down this complex query into independent sub-tasks that AI agents can solve simultaneously.
    Define a list of agents. For each agent, provide a name, a detailed role, the specific task it must accomplish, the tools it might need, and assign a criticality score from 1-10 indicating how complex or important the specific task is.
    The agents will be executed CONCURRENTLY. They cannot depend on each other's outputs. You must make their tasks completely independent.
    Keep the plan as concise as possible but ensure it comprehensively addresses the user's query.
    If the task requires searching the internet, use the tool 'google_search'.
    """

    # JSON-only prompt for models that don't support response_schema
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
                    # Gemma: use plain text with JSON instruction
                    response = client.models.generate_content(
                        model=model,
                        contents=json_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                        ),
                    )
                else:
                    # Gemini: use native structured output
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=AgentPlan,
                        ),
                    )

                raw_text = response.text.strip()

                # For Gemma, extract JSON from possible markdown fences
                if is_gemma and raw_text.startswith("```"):
                    # Strip ```json ... ``` wrapper
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
