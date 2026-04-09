"""
Chat Component — Chat session UI with message display and history.
"""

import streamlit as st
import markdown as md
import html as html_module


def _md_to_html(text: str) -> str:
    """Convert markdown text to HTML for safe embedding inside divs."""
    return md.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )


def init_chat_state():
    """Initialize chat session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_query_result" not in st.session_state:
        st.session_state.current_query_result = None
    if "processing" not in st.session_state:
        st.session_state.processing = False


def render_chat_messages():
    """Render all chat messages in the session."""
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        cached = msg.get("cached", False)
        duration = msg.get("duration", 0)
        model_used = msg.get("model_used", "")

        if role == "user":
            escaped = html_module.escape(content)
            st.markdown(f"""
            <div class="chat-user">
                <div style="font-size: 0.85rem; color: #818cf8; font-weight: 600; margin-bottom: 0.3rem;">
                    👤 You
                </div>
                <div style="font-size: 1rem; line-height: 1.6;">{escaped}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Build header meta badges
            meta_parts = []
            if cached:
                cache_level = msg.get("cache_level", "")
                similarity = msg.get("cache_similarity", 1.0)
                meta_parts.append(
                    f'<span class="chat-badge chat-badge-cache">'
                    f'⚡ CACHED ({html_module.escape(cache_level)}, {similarity:.2f})'
                    f'</span>'
                )
            if duration > 0:
                meta_parts.append(
                    f'<span class="chat-meta">⏱ {duration:.1f}s</span>'
                )
            if model_used:
                meta_parts.append(
                    f'<span class="chat-meta">via {html_module.escape(model_used)}</span>'
                )

            meta_html = " ".join(meta_parts)

            # Convert markdown content → HTML
            content_html = _md_to_html(content)

            st.markdown(f"""
            <div class="chat-assistant">
                <div class="chat-assistant-header">
                    🧠 VibeForge AI {meta_html}
                </div>
                <div class="chat-assistant-content">
                    {content_html}
                </div>
            </div>
            """, unsafe_allow_html=True)


def add_user_message(query: str):
    """Add a user message to the chat."""
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })


def add_assistant_message(
    content: str,
    cached: bool = False,
    cache_level: str = "",
    cache_similarity: float = 1.0,
    duration: float = 0,
    model_used: str = "",
):
    """Add an assistant message to the chat."""
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "cached": cached,
        "cache_level": cache_level,
        "cache_similarity": cache_similarity,
        "duration": duration,
        "model_used": model_used,
    })


def render_history_browser(storage):
    """Render conversation history browser."""
    st.markdown("### 📜 Conversation History")

    records = storage.get_history(limit=20)

    if not records:
        st.info("No conversation history yet. Start chatting!")
        return

    for rec in records:
        query_preview = rec.query[:80] + "..." if len(rec.query) > 80 else rec.query
        with st.expander(
            f"#{rec.id} — {query_preview}  |  {rec.duration_seconds:.1f}s  |  {rec.created_at}",
            expanded=False,
        ):
            st.markdown(f"**Query:** {rec.query}")
            st.markdown(f"**Model:** `{rec.model_used}`")
            st.markdown(f"**Duration:** {rec.duration_seconds:.1f}s")
            st.markdown("---")
            st.markdown(rec.final_answer[:2000] if rec.final_answer else "_No answer_")
