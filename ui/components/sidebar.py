"""
Sidebar Component — BYOK key configuration, settings, and system info.
"""

import streamlit as st


def render_sidebar():
    """Render the sidebar with BYOK, settings, and system status."""
    with st.sidebar:
        # ─── Logo / Branding ──────────────────────────────────────
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
            <div style="font-size: 2.5rem; margin-bottom: 0.2rem;">🧠</div>
            <div class="app-title">VibeForge AI</div>
            <div class="app-subtitle">Multi-Agent Personal Assistant</div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.08); margin: 0.5rem 0 1rem 0;">
        """, unsafe_allow_html=True)

        # ─── API Key Configuration ────────────────────────────────
        st.markdown("### 🔑 API Keys")

        key_mode = st.radio(
            "Key Source",
            ["Default (System Keys)", "BYOK (Bring Your Own)"],
            key="key_mode",
            help="Use system-provided API keys or enter your own Gemini keys"
        )

        if key_mode == "BYOK (Bring Your Own)":
            st.markdown("""
            <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.5rem;">
                Enter your Gemini API keys below. At least one key is required.
            </div>
            """, unsafe_allow_html=True)

            show_keys = st.toggle("Show keys", value=False, key="show_keys")
            input_type = "default" if show_keys else "password"

            byok_keys = {}
            for i in range(1, 5):
                key = st.text_input(
                    f"API Key {i}",
                    value=st.session_state.get(f"byok_key_{i}", ""),
                    type=input_type,
                    key=f"byok_input_{i}",
                    placeholder=f"AIzaSy... (Key {i})"
                )
                if key.strip():
                    byok_keys[i] = key.strip()
                    st.session_state[f"byok_key_{i}"] = key.strip()

            if byok_keys:
                st.success(f"✔ {len(byok_keys)} BYOK key(s) configured")
                st.session_state["byok_keys"] = byok_keys
                st.session_state["use_byok"] = True
            else:
                st.warning("⚠ No BYOK keys entered")
                st.session_state["use_byok"] = False
        else:
            st.session_state["use_byok"] = False
            st.info("Using 4 system-provided API keys")

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        # ─── System Status ────────────────────────────────────────
        st.markdown("### ⚡ System Status")

        pool_status = st.session_state.get("pool_status", {})
        total = pool_status.get("total_slots", 0)
        busy = pool_status.get("busy_slots", 0)
        free = pool_status.get("free_slots", 0)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Slots</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{free}</div>
                <div class="metric-label">Free Slots</div>
            </div>
            """, unsafe_allow_html=True)

        # Cache stats
        cache_stats = st.session_state.get("cache_stats", {})
        hit_rate = cache_stats.get("hit_rate_pct", 0)

        st.markdown(f"""
        <div style="margin-top: 0.8rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8;">
                <span>Cache Hit Rate</span>
                <span style="color: #10b981; font-weight: 600;">{hit_rate:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(hit_rate / 100, 1.0))

        cache_l1 = cache_stats.get("l1_entries", 0)
        cache_l2 = cache_stats.get("l2_entries", 0)
        st.markdown(f"""
        <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.3rem;">
            L1: {cache_l1} entries &nbsp;|&nbsp; L2: {cache_l2} entries
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        # ─── Model Tiers Info ─────────────────────────────────────
        with st.expander("🏗️ Model Architecture", expanded=False):
            st.markdown("""
            <div style="font-size: 0.8rem; line-height: 1.8;">
                <span class="tier-badge tier-0">T0</span> Gemini 3.x — Reasoning & Agentic<br>
                <span class="tier-badge tier-1">T1</span> Gemini 2.5 — General Purpose<br>
                <span class="tier-badge tier-2">T2</span> Gemma 4 — Thinking & Tool Use<br>
                <span class="tier-badge tier-3">T3</span> Gemma 3 — High-Throughput Tasks
            </div>
            """, unsafe_allow_html=True)

        # ─── Pipeline Info ────────────────────────────────────────
        with st.expander("🔄 Pipeline Architecture", expanded=False):
            st.markdown("""
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.6;">
                <b>1. Planner</b> — BM25 RAG → Task decomposition<br>
                <b>2. Executor</b> — Vector RAG → Concurrent agents<br>
                <b>3. Judge</b> — Hybrid RAG → Synthesis & evaluation<br><br>
                <b>Concurrency:</b> Semaphore-based mutex<br>
                <b>Routing:</b> Criticality → Tier mapping<br>
                <b>Fallback:</b> Auto tier demotion on failure<br>
                <b>Cache:</b> L1 SHA-256 + L2 Semantic
            </div>
            """, unsafe_allow_html=True)

        # ─── Footer ──────────────────────────────────────────────
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0 0.5rem 0; font-size: 0.75rem; color: #475569;">
            Built with Gemini API + Semaphores + RAG
        </div>
        """, unsafe_allow_html=True)
