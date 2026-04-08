"""
VibeForge AI — Multi-Agent Personal Assistant

Premium Streamlit frontend for the multi-agent pipeline.
Features:
  - Chat session with persistent history
  - Real-time agent execution visualization
  - BYOK (Bring Your Own Key) sidebar
  - System metrics dashboard
  - Conversation history browser
"""

import sys
import json
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ─── Page Config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="VibeForge AI — Multi-Agent Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Custom CSS ─────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ─── Imports (after sys.path setup) ──────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from ui.components.sidebar import render_sidebar
from ui.components.chat import (
    init_chat_state,
    render_chat_messages,
    add_user_message,
    add_assistant_message,
)
from ui.components.agent_viz import (
    render_pipeline_status,
    render_plan_agents,
    render_agent_outputs,
    render_event_log,
)
from ui.components.metrics import render_metrics_dashboard


# ─── Initialize Core Systems (cached) ────────────────────────────────────────

@st.cache_resource(show_spinner="🚀 Initializing Multi-Agent Engine...")
def init_core_systems():
    """Initialize all core systems once. Cached across reruns."""
    from core.llm import get_clients
    from core.resource_pool import ResourcePool
    from core.rag_engine import initialize_rag
    from core.storage import StorageManager
    from core.cache import QueryCache

    clients = get_clients()
    pool = ResourcePool(clients)
    initialize_rag()
    storage = StorageManager()
    cache = QueryCache()

    return clients, pool, storage, cache


def get_byok_clients_and_pool(byok_keys: dict):
    """Create temporary clients and pool from BYOK keys."""
    from core.llm import get_clients
    from core.resource_pool import ResourcePool

    clients = get_clients(api_keys=byok_keys)
    pool = ResourcePool(clients)
    return clients, pool


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    # Initialize session state
    init_chat_state()

    # Initialize core (cached)
    try:
        default_clients, default_pool, storage, cache = init_core_systems()
        system_ready = True
    except Exception as e:
        system_ready = False
        st.error(f"❌ Failed to initialize: {e}")
        st.info("Check your `.env` file has valid GEMINI_API_KEY1-4.")
        return

    # Update session state with current status
    st.session_state["pool_status"] = default_pool.get_pool_status()
    st.session_state["cache_stats"] = cache.stats()

    # ─── Sidebar ──────────────────────────────────────────────────
    render_sidebar()

    # ─── Resolve active clients/pool (BYOK or default) ────────────
    if st.session_state.get("use_byok") and st.session_state.get("byok_keys"):
        try:
            clients, pool = get_byok_clients_and_pool(st.session_state["byok_keys"])
        except Exception:
            clients, pool = default_clients, default_pool
    else:
        clients, pool = default_clients, default_pool

    # ─── Header ───────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.02em;">
            🧠 VibeForge AI
        </div>
        <div style="font-size: 0.9rem; color: #64748b; margin-top: -0.3rem;">
            Multi-Agent Personal Assistant — Powered by Gemini + Semaphore Concurrency + RAG
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Tabs ─────────────────────────────────────────────────────
    tab_chat, tab_agents, tab_history, tab_metrics = st.tabs([
        "💬 Chat", "🤖 Agents", "📜 History", "📊 Metrics"
    ])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1: CHAT
    # ═══════════════════════════════════════════════════════════════
    with tab_chat:
        # Render existing messages
        render_chat_messages()

        # Query input
        query = st.chat_input(
            "Ask me anything — I'll deploy a team of AI agents...",
            key="chat_input",
        )

        if query and query.strip():
            # Add user message
            add_user_message(query.strip())

            # Check cache first
            cached = cache.lookup(query.strip())
            if cached is not None:
                add_assistant_message(
                    content=cached.cached_answer,
                    cached=True,
                    cache_level=cached.cache_level,
                    cache_similarity=cached.similarity,
                )
                st.session_state["pipeline_stage"] = "complete"
                st.rerun()

            # Run full pipeline
            with st.status("🚀 Running multi-agent pipeline...", expanded=True) as status:
                # Phase 1: Planning
                status.update(label="🧩 Planner Agent is designing the solution...")
                st.session_state["pipeline_stage"] = "planning"

                try:
                    from core.planner import build_plan
                    plan = build_plan(clients, query.strip())
                except Exception as e:
                    st.error(f"❌ Planning failed: {e}")
                    st.session_state["pipeline_stage"] = "idle"
                    st.stop()

                # Store plan in session
                plan_data = [
                    {
                        "name": a.name,
                        "role": a.role,
                        "task": a.task[:500],
                        "tools_required": a.tools_required,
                        "criticality": a.criticality,
                    }
                    for a in plan.agents
                ]
                st.session_state["last_plan"] = plan_data

                st.markdown(f"**✅ Plan created: {len(plan.agents)} agents**")
                for i, a in enumerate(plan.agents):
                    st.markdown(f"  {i+1}. **{a.name}** (crit={a.criticality})")

                # Phase 2: Execution
                status.update(label="⚡ Agents executing across resource pool...")
                st.session_state["pipeline_stage"] = "executing"

                agent_outputs_list = []
                from core.executor import execute_plan
                final_context = execute_plan(
                    clients, plan, pool, agent_outputs=agent_outputs_list
                )
                st.session_state["last_agent_outputs"] = agent_outputs_list

                st.markdown(f"**✅ {len(agent_outputs_list)} agents completed**")

                if not final_context.strip():
                    st.error("❌ Agents failed to produce any context.")
                    st.session_state["pipeline_stage"] = "idle"
                    st.stop()

                # Phase 3: Judge
                status.update(label="⚖️ Judge Agent is synthesizing...")
                st.session_state["pipeline_stage"] = "judging"

                t_start = time.time()
                from core.judge import evaluate_and_synthesize
                final_answer, model_used = evaluate_and_synthesize(
                    clients, pool, query.strip(), final_context
                )
                duration = time.time() - t_start

                if final_answer.strip():
                    # Persist
                    plan_json = json.dumps({"agents": plan_data}, indent=2)
                    storage.save_conversation(
                        query=query.strip(),
                        plan_json=plan_json,
                        agent_outputs=agent_outputs_list,
                        final_answer=final_answer,
                        model_used=model_used,
                        duration_seconds=duration,
                    )
                    cache.store(query.strip(), final_answer)

                    from core.rag_engine import add_user_interaction
                    add_user_interaction(query.strip(), final_answer)

                    # Add to chat
                    add_assistant_message(
                        content=final_answer,
                        duration=duration,
                        model_used=model_used,
                    )

                    # Store event log
                    st.session_state["last_event_log"] = pool.get_event_log()

                    status.update(label="✅ Complete!", state="complete")
                    st.session_state["pipeline_stage"] = "complete"
                else:
                    st.error("❌ Judge failed to produce an answer.")
                    st.session_state["pipeline_stage"] = "idle"

            # Update stats
            st.session_state["pool_status"] = pool.get_pool_status()
            st.session_state["cache_stats"] = cache.stats()
            st.rerun()

    # ═══════════════════════════════════════════════════════════════
    # TAB 2: AGENTS
    # ═══════════════════════════════════════════════════════════════
    with tab_agents:
        stage = st.session_state.get("pipeline_stage", "idle")
        render_pipeline_status(stage)

        st.markdown("---")

        agent_col, log_col = st.columns([3, 2])

        with agent_col:
            # Show plan agents
            last_plan = st.session_state.get("last_plan", [])
            if last_plan:
                render_plan_agents(last_plan)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; color: #475569;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
                    <div style="font-size: 1.1rem; font-weight: 600;">No Active Plan</div>
                    <div style="font-size: 0.85rem; margin-top: 0.3rem;">
                        Send a query in the Chat tab to see agents in action
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Show agent outputs
            last_outputs = st.session_state.get("last_agent_outputs", [])
            if last_outputs:
                st.markdown("---")
                render_agent_outputs(last_outputs)

        with log_col:
            event_log = st.session_state.get("last_event_log", [])
            render_event_log(event_log)

    # ═══════════════════════════════════════════════════════════════
    # TAB 3: HISTORY
    # ═══════════════════════════════════════════════════════════════
    with tab_history:
        from ui.components.chat import render_history_browser
        render_history_browser(storage)

    # ═══════════════════════════════════════════════════════════════
    # TAB 4: METRICS
    # ═══════════════════════════════════════════════════════════════
    with tab_metrics:
        pool_status = pool.get_pool_status()
        cache_stats_data = cache.stats()
        storage_info = {
            "total_conversations": len(storage.get_history(limit=9999)),
            "user_kb_docs": storage.get_user_kb_count(),
            "rag_initialized": True,
        }
        render_metrics_dashboard(pool_status, cache_stats_data, storage_info)


if __name__ == "__main__":
    main()
