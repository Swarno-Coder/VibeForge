"""
LLM Client Factory — creates multiple genai.Client instances,
one per API key (GEMINI_API_KEY1 through GEMINI_API_KEY4).

Supports BYOK (Bring Your Own Key) for API/UI usage.
"""

import os
from google import genai
from rich.console import Console

console = Console()


def get_clients(api_keys: dict[int, str] | None = None) -> dict[int, genai.Client]:
    """
    Build and return a dictionary of {key_index: genai.Client}.

    Args:
        api_keys: Optional dict {1: "key1", 2: "key2", ...} for BYOK.
                  If None, reads GEMINI_API_KEY1..N from environment.

    Raises ValueError if no keys are found at all.
    """
    clients: dict[int, genai.Client] = {}

    if api_keys:
        # BYOK mode: use supplied keys
        for idx, key in api_keys.items():
            if key and key.strip():
                clients[idx] = genai.Client(api_key=key.strip())
                console.print(f"[dim]✔ Loaded BYOK key #{idx}[/dim]")
    else:
        # Default mode: read from environment
        num_keys = int(os.getenv("NUM_KEYS", "4"))
        for i in range(1, num_keys + 1):
            env_var = f"GEMINI_API_KEY{i}"
            api_key = os.getenv(env_var)
            if api_key:
                clients[i] = genai.Client(api_key=api_key)
                console.print(f"[dim]✔ Loaded {env_var}[/dim]")
            else:
                console.print(f"[bold yellow]⚠ {env_var} not set — skipping[/bold yellow]")

    if not clients:
        raise ValueError(
            "No API keys found. Set GEMINI_API_KEY1 through GEMINI_API_KEY4 in .env "
            "or provide keys via BYOK."
        )

    console.print(f"[bold green]Loaded {len(clients)} API key(s)[/bold green]")
    return clients
