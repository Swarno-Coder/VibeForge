"""
LLM Client Factory — creates multiple genai.Client instances,
one per API key (GEMINI_API_KEY1 through GEMINI_API_KEY4).
"""

import os
from google import genai
from rich.console import Console

console = Console()

NUM_KEYS = os.getenv("NUM_KEYS", 4)


def get_clients() -> dict[int, genai.Client]:
    """
    Build and return a dictionary of {key_index: genai.Client}.
    Reads GEMINI_API_KEY1 .. GEMINI_API_KEY4 from environment.

    Raises ValueError if no keys are found at all.
    """
    clients: dict[int, genai.Client] = {}

    for i in range(1, NUM_KEYS + 1):
        env_var = f"GEMINI_API_KEY{i}"
        api_key = os.getenv(env_var)
        if api_key:
            clients[i] = genai.Client(api_key=api_key)
            console.print(f"[dim]✔ Loaded {env_var}[/dim]")
        else:
            console.print(f"[bold yellow]⚠ {env_var} not set — skipping[/bold yellow]")

    if not clients:
        raise ValueError(
            "No API keys found. Set GEMINI_API_KEY1 through GEMINI_API_KEY4 in .env"
        )

    console.print(f"[bold green]Loaded {len(clients)} API key(s)[/bold green]")
    return clients
