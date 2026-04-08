"""
Resource Pool with Semaphore-based mutual exclusion and HashMap allocation tracking.

Architecture:
  - 4 API keys × 12 models = 48 resource slots
  - Semaphore(48) enforces mutual exclusion — at most 48 concurrent calls
  - HashMap (allocation_map) logs every ALLOCATE / DEALLOCATE event in real-time
  - 4-tier criticality-based model routing
  - Retry logic falls back across API keys and model tiers on error / bad JSON
  - Rate limit metadata per model for awareness

Model Ranking (best → lightest):
  Tier 0 — Gemini 3.x  (thinking + reasoning + agentic)
  Tier 1 — Gemini 2.5   (solid general purpose)
  Tier 2 — Gemma 4      (thinking + agentic, generous rate limits)
  Tier 3 — Gemma 3      (high rate limits, good for repetitive low-intelligence tasks)
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional
from google import genai
from google.genai import types
from rich.console import Console
from rich.table import Table

console = Console()

# ─── Model Tiers (best → lightest) ──────────────────────────────────────────
MODEL_TIERS = [
    # Tier 0 — Gemini 3.x (Criticality 9-10)
    [
        ("gemini-3.1-flash-lite-preview", 15),
        ("gemini-3-flash-preview", 5),
    ],
    # Tier 1 — Gemini 2.5 (Criticality 7-8)
    [
        ("gemini-2.5-flash", 5),
        ("gemini-2.5-flash-lite", 10),
    ],
    # Tier 2 — Gemma 4 (Criticality 4-6)
    [
        ("gemma-4-31b-it", 15),
        ("gemma-4-26b-a4b-it", 15),
    ],
    # Tier 3 — Gemma 3 (Criticality 1-3)
    [
        ("gemma-3-27b-it", 30),
        ("gemma-3-12b-it", 30),
        ("gemma-3-4b-it", 30),
        ("gemma-3n-e4b-it", 30),
        ("gemma-3n-e2b-it", 30),
        ("gemma-3-1b-it", 30),
    ],
]

# Flat list for convenience
ALL_MODELS = []
MODEL_RATE_LIMITS = {}
for tier in MODEL_TIERS:
    for model_name, rpm in tier:
        ALL_MODELS.append(model_name)
        MODEL_RATE_LIMITS[model_name] = rpm


def _criticality_to_start_tier(criticality: int) -> int:
    """Map agent criticality (1-10) to starting tier index."""
    if criticality >= 9:
        return 0   # Gemini 3.x
    elif criticality >= 7:
        return 1   # Gemini 2.5
    elif criticality >= 4:
        return 2   # Gemma 4
    else:
        return 3   # Gemma 3


@dataclass
class ResourceSlot:
    """Represents a single (api_key_index, model) resource slot."""
    slot_id: str          # e.g. "KEY1_gemini-3.1-flash-lite-preview"
    key_index: int        # 1-4
    model: str            # model name string
    client: genai.Client  # pre-built genai client for this key
    rate_limit: int       # RPM per this key for this model
    tier: int             # which tier (0-3) this model belongs to
    busy: bool = False    # guarded by ResourcePool._map_lock


class ResourcePool:
    """
    Thread-safe resource pool using:
      - Semaphore for mutual exclusion of resource slots
      - HashMap (dict) for real-time allocation logging
    """

    def __init__(self, clients: dict[int, genai.Client]):
        # Build all slots: 4 keys × 12 models = 48 slots
        self._slots: list[ResourceSlot] = []
        for key_idx, client in clients.items():
            for tier_idx, tier_models in enumerate(MODEL_TIERS):
                for model_name, rpm in tier_models:
                    slot_id = f"KEY{key_idx}_{model_name}"
                    self._slots.append(ResourceSlot(
                        slot_id=slot_id,
                        key_index=key_idx,
                        model=model_name,
                        client=client,
                        rate_limit=rpm,
                        tier=tier_idx,
                        busy=False,
                    ))

        total_slots = len(self._slots)

        # ─── Semaphore: limits concurrency to total available slots ───
        self._semaphore = threading.Semaphore(total_slots)

        # ─── HashMap: tracks live allocations ─────────────────────────
        self._allocation_map: dict[str, dict] = {}   # slot_id → metadata
        self._map_lock = threading.Lock()             # protects HashMap + slot.busy

        # ─── Event log: full history ──────────────────────────────────
        self._event_log: list[dict] = []
        self._log_lock = threading.Lock()

        # ─── Temporarily blacklisted slots (auto-clears after TTL) ────
        self._blacklist: dict[str, float] = {}  # slot_id → expiry timestamp
        self._blacklist_ttl = 60  # seconds

        tier_names = ["Gemini 3.x", "Gemini 2.5", "Gemma 4", "Gemma 3"]
        console.print(f"\n[bold green]ResourcePool initialised: {total_slots} slots "
                      f"({len(clients)} keys × {len(ALL_MODELS)} models)[/bold green]")
        for t_idx, tier_models in enumerate(MODEL_TIERS):
            models_str = ", ".join(f"{m}({r}rpm)" for m, r in tier_models)
            console.print(f"  [dim]Tier {t_idx} ({tier_names[t_idx]}): {models_str}[/dim]")

    # ─── Core: Acquire ────────────────────────────────────────────────────────

    def acquire(self, agent_name: str, criticality: int) -> Optional[ResourceSlot]:
        acquired = self._semaphore.acquire(timeout=120)
        if not acquired:
            console.print(f"[bold red]Semaphore timeout for {agent_name}[/bold red]")
            return None

        start_tier = _criticality_to_start_tier(criticality)
        now = time.time()
        num_tiers = len(MODEL_TIERS)

        with self._map_lock:
            tier_order = (
                list(range(start_tier, num_tiers)) +
                list(range(0, start_tier))
            )

            for tier_idx in tier_order:
                for slot in self._slots:
                    if slot.tier == tier_idx and not slot.busy:
                        if slot.slot_id in self._blacklist:
                            if now < self._blacklist[slot.slot_id]:
                                continue
                            else:
                                del self._blacklist[slot.slot_id]

                        slot.busy = True
                        self._allocation_map[slot.slot_id] = {
                            "agent": agent_name,
                            "model": slot.model,
                            "key_index": slot.key_index,
                            "tier": tier_idx,
                            "rate_limit": slot.rate_limit,
                            "acquired_at": time.strftime("%H:%M:%S"),
                            "timestamp": now,
                        }
                        self._log_event("ALLOCATE", agent_name, slot)
                        return slot

            self._semaphore.release()
            console.print(
                f"[bold yellow]No suitable slot for {agent_name} "
                f"(crit={criticality}), all blacklisted or busy[/bold yellow]"
            )
            return None

    # ─── Core: Release ────────────────────────────────────────────────────────

    def release(self, slot: ResourceSlot, agent_name: str):
        with self._map_lock:
            slot.busy = False
            if slot.slot_id in self._allocation_map:
                del self._allocation_map[slot.slot_id]
            self._log_event("DEALLOCATE", agent_name, slot)
        self._semaphore.release()

    # ─── Blacklist a slot temporarily ─────────────────────────────────────────

    def blacklist_slot(self, slot: ResourceSlot):
        with self._map_lock:
            self._blacklist[slot.slot_id] = time.time() + self._blacklist_ttl

    # ─── HashMap: live allocation table ───────────────────────────────────────

    def print_allocation_table(self):
        with self._map_lock:
            if not self._allocation_map:
                console.print("[dim]  (no active allocations)[/dim]")
                return

            table = Table(
                title="🔒 Live Resource Allocation (HashMap)",
                show_lines=True,
                title_style="bold cyan",
            )
            table.add_column("Slot ID", style="cyan", min_width=35)
            table.add_column("Agent", style="green")
            table.add_column("Model", style="yellow")
            table.add_column("Key#", style="magenta", justify="center")
            table.add_column("Tier", style="white", justify="center")
            table.add_column("RPM", style="dim", justify="center")
            table.add_column("Acquired", style="white")

            tier_names = {0: "T0-Gem3", 1: "T1-Gem2.5", 2: "T2-Gma4", 3: "T3-Gma3"}
            for slot_id, info in self._allocation_map.items():
                table.add_row(
                    slot_id,
                    info["agent"],
                    info["model"],
                    str(info["key_index"]),
                    tier_names.get(info["tier"], str(info["tier"])),
                    str(info["rate_limit"]),
                    info["acquired_at"],
                )

            console.print(table)

    def get_allocation_snapshot(self) -> dict:
        with self._map_lock:
            return dict(self._allocation_map)

    # ─── Status getters for API/UI ────────────────────────────────────────────

    def get_pool_status(self) -> dict:
        """Return pool status dict for API/UI consumption."""
        with self._map_lock:
            busy_count = sum(1 for s in self._slots if s.busy)
            blacklisted_count = len([
                sid for sid, exp in self._blacklist.items()
                if time.time() < exp
            ])
            return {
                "total_slots": len(self._slots),
                "busy_slots": busy_count,
                "free_slots": len(self._slots) - busy_count,
                "blacklisted_slots": blacklisted_count,
                "active_allocations": len(self._allocation_map),
                "allocation_map": dict(self._allocation_map),
            }

    def get_event_log(self) -> list[dict]:
        """Return a copy of the full event log for API/UI."""
        with self._log_lock:
            return list(self._event_log)

    # ─── Event log ────────────────────────────────────────────────────────────

    def _log_event(self, action: str, agent_name: str, slot: ResourceSlot):
        tier_labels = {0: "Gemini3.x", 1: "Gemini2.5", 2: "Gemma4", 3: "Gemma3"}
        event = {
            "time": time.strftime("%H:%M:%S"),
            "action": action,
            "agent": agent_name,
            "slot_id": slot.slot_id,
            "model": slot.model,
            "key_index": slot.key_index,
            "tier": slot.tier,
            "rate_limit": slot.rate_limit,
        }
        with self._log_lock:
            self._event_log.append(event)

        icon = "🔒" if action == "ALLOCATE" else "🔓"
        color = "green" if action == "ALLOCATE" else "blue"
        tier_label = tier_labels.get(slot.tier, f"T{slot.tier}")
        console.print(
            f"[bold {color}]{icon} {action}[/bold {color}] "
            f"[dim]{event['time']}[/dim] "
            f"{agent_name} ↔ KEY{slot.key_index}/{slot.model} "
            f"[dim]({tier_label}, {slot.rate_limit}rpm)[/dim]"
        )

    def save_log_to_file(self, path: str = "allocation_log.txt"):
        tier_labels = {0: "Gemini3.x", 1: "Gemini2.5", 2: "Gemma4", 3: "Gemma3"}
        with self._log_lock:
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    f"{'TIME':<12} {'ACTION':<14} {'AGENT':<25} "
                    f"{'SLOT':<45} {'KEY#':<6} {'TIER':<12} {'RPM':<6} {'MODEL'}\n"
                )
                f.write("=" * 140 + "\n")
                for evt in self._event_log:
                    f.write(
                        f"{evt['time']:<12} {evt['action']:<14} {evt['agent']:<25} "
                        f"{evt['slot_id']:<45} {evt['key_index']:<6} "
                        f"{tier_labels.get(evt['tier'], 'T?'):<12} "
                        f"{evt['rate_limit']:<6} {evt['model']}\n"
                    )
            console.print(f"[dim]Allocation log saved to {path}[/dim]")

    # ─── Blacklist ALL keys for a given model ─────────────────────────────────

    def blacklist_model(self, model_name: str):
        with self._map_lock:
            expiry = time.time() + self._blacklist_ttl
            for slot in self._slots:
                if slot.model == model_name:
                    self._blacklist[slot.slot_id] = expiry
            console.print(
                f"[dim]🚫 Blacklisted ALL keys for model {model_name} "
                f"for {self._blacklist_ttl}s[/dim]"
            )

    # ─── Execute with retry ───────────────────────────────────────────────────

    def execute_with_retry(
        self,
        agent,
        max_retries: int = 12,
        agent_outputs: list | None = None,
    ) -> str:
        """
        Execute an agent task with robust retry + tier demotion logic.

        Args:
            agent_outputs: Optional list to append per-agent result dicts
                           for API/UI consumption.
        """
        agent_name = agent.name
        original_criticality = getattr(agent, "criticality", 5)
        effective_criticality = original_criticality
        last_error = None

        tier_fail_counts: dict[int, int] = {}
        FAILURES_BEFORE_DEMOTE = 2
        num_tiers = len(MODEL_TIERS)

        tier_crit_map = {
            0: 9,
            1: 7,
            2: 4,
            3: 1,
        }

        for attempt in range(1, max_retries + 1):
            slot = self.acquire(agent_name, effective_criticality)
            if slot is None:
                console.print(
                    f"[bold yellow]⚠ {agent_name}: No slot available "
                    f"(attempt {attempt}/{max_retries}, eff_crit={effective_criticality}), "
                    f"waiting...[/bold yellow]"
                )
                if effective_criticality > 1:
                    effective_criticality = 1
                    console.print(
                        f"[bold magenta]⬇ {agent_name}: emergency demote to "
                        f"Tier 3 (Gemma 3)[/bold magenta]"
                    )
                time.sleep(2)
                continue

            try:
                wants_search = "google_search" in agent.tools_required
                stable_grounding_models = {
                    "gemini-2.5-flash", "gemini-2.5-flash-lite",
                }
                can_ground = slot.model in stable_grounding_models

                config = types.GenerateContentConfig(temperature=0.4)
                if wants_search and can_ground:
                    search_tool = types.Tool(google_search=types.GoogleSearch())
                    config.tools = [search_tool]

                prompt = (
                    f"You are {agent.name}.\n"
                    f"Your Role: {agent.role}\n"
                    f"Your Mission: {agent.task}\n"
                    f"Complete your mission thoroughly."
                )
                if wants_search and not can_ground:
                    prompt += (
                        "\n\nNote: You do not have access to a live search tool. "
                        "Use your training knowledge to provide the best possible answer."
                    )

                tier_labels = {0: "Gemini3.x", 1: "Gemini2.5", 2: "Gemma4", 3: "Gemma3"}
                tier_label = tier_labels.get(slot.tier, f"T{slot.tier}")
                grounding_note = "+search" if (wants_search and can_ground) else ""
                console.print(
                    f"[dim]▶ {agent_name} → KEY{slot.key_index}/{slot.model} "
                    f"({tier_label}, {slot.rate_limit}rpm{grounding_note}) "
                    f"[attempt {attempt}/{max_retries}, crit={original_criticality}"
                    f"→eff={effective_criticality}][/dim]"
                )

                response = slot.client.models.generate_content(
                    model=slot.model,
                    contents=prompt,
                    config=config,
                )

                if not response.text or not response.text.strip():
                    raise ValueError("Empty response from model")

                result = (
                    f"\n\n--- Output from {agent_name} "
                    f"(via KEY{slot.key_index}/{slot.model}, {tier_label}) ---\n"
                    f"{response.text}"
                )

                # Collect per-agent output for API/UI
                if agent_outputs is not None:
                    agent_outputs.append({
                        "agent_name": agent_name,
                        "content": response.text,
                        "model": slot.model,
                        "tier": slot.tier,
                        "tier_label": tier_label,
                        "key_index": slot.key_index,
                        "criticality": original_criticality,
                        "attempts": attempt,
                    })

                demote_note = ""
                if slot.tier != _criticality_to_start_tier(original_criticality):
                    demote_note = f" [demoted from Tier {_criticality_to_start_tier(original_criticality)}]"
                console.print(
                    f"[bold green]✔ {agent_name} completed on "
                    f"KEY{slot.key_index}/{slot.model} ({tier_label}){demote_note}[/bold green]"
                )
                return result

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                is_quota = any(
                    k in error_str
                    for k in ["429", "resource_exhausted", "quota"]
                )
                is_json_error = any(
                    k in error_str
                    for k in ["json", "parse", "decode", "invalid"]
                )

                err_type = "QUOTA" if is_quota else ("JSON" if is_json_error else "ERROR")
                console.print(
                    f"[bold yellow]⚠ {agent_name} [{err_type}] on "
                    f"KEY{slot.key_index}/{slot.model} (Tier {slot.tier}): {e} "
                    f"(attempt {attempt}/{max_retries})[/bold yellow]"
                )

                if is_quota:
                    self.blacklist_model(slot.model)
                else:
                    self.blacklist_slot(slot)

                failed_tier = slot.tier
                tier_fail_counts[failed_tier] = tier_fail_counts.get(failed_tier, 0) + 1

                if tier_fail_counts[failed_tier] >= FAILURES_BEFORE_DEMOTE:
                    next_tier = failed_tier + 1
                    if next_tier < num_tiers:
                        effective_criticality = tier_crit_map[next_tier]
                        console.print(
                            f"[bold magenta]⬇ {agent_name}: {FAILURES_BEFORE_DEMOTE} "
                            f"failures on Tier {failed_tier} → demoting to "
                            f"Tier {next_tier} ({tier_labels.get(next_tier, '?')}), "
                            f"eff_crit={effective_criticality}[/bold magenta]"
                        )
                    else:
                        console.print(
                            f"[bold red]⚠ {agent_name}: already at lowest tier "
                            f"(Tier {failed_tier}), no further demotion possible[/bold red]"
                        )

                if attempt < max_retries:
                    wait_time = 2.0 if is_quota else 0.5
                    time.sleep(wait_time)

            finally:
                self.release(slot, agent_name)

        console.print(
            f"[bold red]✘ {agent_name} FAILED after {max_retries} attempts "
            f"across all tiers. Last error: {last_error}[/bold red]"
        )
        return (
            f"\n\n--- Output from {agent_name} ---\n"
            f"[FAILED after {max_retries} attempts across all tiers: {last_error}]"
        )
