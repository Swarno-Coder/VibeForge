"""
Metrics Component — System metrics dashboard.
"""

import streamlit as st


def render_metrics_dashboard(pool_status: dict, cache_stats: dict, storage_info: dict):
    """Render the system metrics dashboard."""
    st.markdown("### 📊 System Metrics")

    # ─── Top-Level Metrics Row ────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pool_status.get('total_slots', 0)}</div>
            <div class="metric-label">Resource Slots</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pool_status.get('free_slots', 0)}</div>
            <div class="metric-label">Available</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        hit_rate = cache_stats.get("hit_rate_pct", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{hit_rate:.0f}%</div>
            <div class="metric-label">Cache Hit Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_convs = storage_info.get("total_conversations", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_convs}</div>
            <div class="metric-label">Conversations</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Detailed Stats ──────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top: 0;">🔒 Resource Pool</h4>
        </div>
        """, unsafe_allow_html=True)

        pool_data = {
            "Metric": ["Total Slots", "Busy", "Free", "Blacklisted", "Active Allocs"],
            "Value": [
                pool_status.get("total_slots", 0),
                pool_status.get("busy_slots", 0),
                pool_status.get("free_slots", 0),
                pool_status.get("blacklisted_slots", 0),
                pool_status.get("active_allocations", 0),
            ],
        }
        st.table(pool_data)

    with col_right:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top: 0;">💾 Cache Statistics</h4>
        </div>
        """, unsafe_allow_html=True)

        cache_data = {
            "Metric": ["L1 Entries", "L2 Entries", "L1 Hits", "L2 Hits", "Misses", "Hit Rate"],
            "Value": [
                str(cache_stats.get("l1_entries", 0)),
                str(cache_stats.get("l2_entries", 0)),
                str(cache_stats.get("l1_hits", 0)),
                str(cache_stats.get("l2_hits", 0)),
                str(cache_stats.get("misses", 0)),
                f"{cache_stats.get('hit_rate_pct', 0):.1f}%",
            ],
        }
        st.table(cache_data)

    # ─── Knowledge Base Stats ─────────────────────────────────────
    st.markdown("""
    <div class="glass-card" style="margin-top: 1rem;">
        <h4 style="margin-top: 0;">📚 Knowledge Base & Storage</h4>
    </div>
    """, unsafe_allow_html=True)

    kb_col1, kb_col2, kb_col3 = st.columns(3)
    with kb_col1:
        st.metric("Conversations", storage_info.get("total_conversations", 0))
    with kb_col2:
        st.metric("User KB Docs", storage_info.get("user_kb_docs", 0))
    with kb_col3:
        st.metric("RAG Status", "✅ Active" if storage_info.get("rag_initialized", False) else "❌ Inactive")
