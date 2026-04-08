"""
Agent Visualization Component — Real-time display of agent execution pipeline.
"""

import streamlit as st


def render_pipeline_status(stage: str = "idle"):
    """Render the pipeline flow indicator: Planner → Executor → Judge."""
    stages = {
        "idle": (False, False, False),
        "planning": (True, False, False),
        "executing": (False, True, False),
        "judging": (False, False, True),
        "complete": (False, False, False),
    }
    plan_active, exec_active, judge_active = stages.get(stage, (False, False, False))

    def step_class(active, complete=False):
        if active:
            return "active"
        if stage == "complete" or complete:
            return "complete"
        return ""

    plan_cls = step_class(plan_active, stage in ["executing", "judging", "complete"])
    exec_cls = step_class(exec_active, stage in ["judging", "complete"])
    judge_cls = step_class(judge_active, stage == "complete")

    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; padding: 1rem 0; flex-wrap: wrap;">
        <div class="pipeline-step {plan_cls}">
            🧩 Planner
        </div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step {exec_cls}">
            ⚡ Executor
        </div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step {judge_cls}">
            ⚖️ Judge
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_plan_agents(plan_agents: list[dict]):
    """Render agent cards from the execution plan."""
    if not plan_agents:
        return

    st.markdown("#### 🤖 Agent Deployment Plan")

    for i, agent in enumerate(plan_agents):
        name = agent.get("name", f"Agent {i+1}")
        role = agent.get("role", "")
        task = agent.get("task", "")
        crit = agent.get("criticality", 5)
        tools = agent.get("tools_required", [])

        # Criticality color
        if crit >= 9:
            crit_color = "#6366f1"
            tier_label = "T0"
        elif crit >= 7:
            crit_color = "#06b6d4"
            tier_label = "T1"
        elif crit >= 4:
            crit_color = "#10b981"
            tier_label = "T2"
        else:
            crit_color = "#f59e0b"
            tier_label = "T3"

        tools_str = ", ".join(tools) if tools else "none"

        st.markdown(f"""
        <div class="agent-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <span style="font-weight: 600; font-size: 0.95rem;">{name}</span>
                <div>
                    <span class="tier-badge tier-{min(3, max(0, 3 - (crit - 1) // 3))}">{tier_label}</span>
                    <span style="font-size: 0.8rem; color: {crit_color}; font-weight: 600; margin-left: 0.5rem;">
                        Crit: {crit}/10
                    </span>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.3rem;">
                {role[:150]}{'...' if len(role) > 150 else ''}
            </div>
            <div style="font-size: 0.78rem; color: #64748b;">
                🎯 {task[:200]}{'...' if len(task) > 200 else ''}
            </div>
            <div style="font-size: 0.75rem; color: #475569; margin-top: 0.3rem;">
                🔧 Tools: {tools_str}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_agent_outputs(agent_outputs: list[dict]):
    """Render individual agent execution results."""
    if not agent_outputs:
        return

    st.markdown("#### 📋 Agent Execution Results")

    for ao in agent_outputs:
        name = ao.get("agent_name", "Unknown")
        model = ao.get("model", "")
        tier_label = ao.get("tier_label", "")
        attempts = ao.get("attempts", 1)
        content = ao.get("content", "")

        st.markdown(f"""
        <div class="agent-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                <span style="font-weight: 600;">✅ {name}</span>
                <span style="font-size: 0.78rem; color: #64748b;">
                    {model} ({tier_label}) · {attempts} attempt(s)
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"View {name}'s output", expanded=False):
            st.markdown(content[:3000])


def render_event_log(events: list[dict]):
    """Render the resource allocation event log."""
    if not events:
        st.info("No allocation events yet.")
        return

    st.markdown("#### 📜 Resource Allocation Log")

    # Show last 20 events
    recent = events[-20:]
    for evt in reversed(recent):
        action = evt.get("action", "")
        icon = "🔒" if action == "ALLOCATE" else "🔓"
        color = "#10b981" if action == "ALLOCATE" else "#06b6d4"
        agent = evt.get("agent", "")
        model = evt.get("model", "")
        key_idx = evt.get("key_index", 0)
        time_str = evt.get("time", "")

        st.markdown(f"""
        <div style="font-size: 0.8rem; padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
            <span style="color: {color}; font-weight: 600;">{icon} {action}</span>
            <span style="color: #64748b;"> {time_str}</span>
            <span style="color: #94a3b8;"> {agent}</span>
            <span style="color: #475569;"> → KEY{key_idx}/{model}</span>
        </div>
        """, unsafe_allow_html=True)
