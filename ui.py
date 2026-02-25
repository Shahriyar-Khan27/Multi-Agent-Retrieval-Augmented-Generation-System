import streamlit as st
import os
import sys
import time
from dotenv import load_dotenv

# MUST be the first Streamlit command
st.set_page_config(
    page_title="RAG Agent Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure the project root is in the Python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agentic_rag_assistant import run_agent
from src.utils import ingest_documents

load_dotenv()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Page background ── */
.stApp {
    background: #0f1117;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 1.5rem;
}
.sidebar-logo-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.sidebar-logo-text { line-height: 1.2; }
.sidebar-logo-title {
    font-size: 15px; font-weight: 700;
    color: #e2e8f0; letter-spacing: -0.3px;
}
.sidebar-logo-sub {
    font-size: 11px; color: #64748b; font-weight: 400;
}

/* ── Sidebar section headers ── */
.sidebar-section-title {
    font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
    color: #4b5563; text-transform: uppercase;
    margin: 1.4rem 0 0.6rem 0;
}

/* ── Capability badges ── */
.cap-badge {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 10px;
    background: #1a2035;
    border: 1px solid #1e2535;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 12px; color: #94a3b8;
}
.cap-badge-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Stat cards ── */
.stat-row {
    display: flex; gap: 8px; margin-bottom: 1rem;
}
.stat-card {
    flex: 1;
    background: #1a2035;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 10px 8px;
    text-align: center;
}
.stat-card-value {
    font-size: 20px; font-weight: 700; color: #818cf8;
    line-height: 1;
}
.stat-card-label {
    font-size: 10px; color: #4b5563; margin-top: 3px;
    text-transform: uppercase; letter-spacing: 0.6px;
}

/* ── Main header ── */
.main-header {
    padding: 1.6rem 0 1.2rem 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 1.5rem;
}
.main-header-title {
    font-size: 26px; font-weight: 700;
    color: #e2e8f0; letter-spacing: -0.5px;
    margin: 0;
}
.main-header-sub {
    font-size: 13px; color: #64748b;
    margin-top: 4px;
}
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0d2218;
    border: 1px solid #134d2e;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px; font-weight: 600; color: #4ade80;
}
.status-dot {
    width: 6px; height: 6px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Chat container ── */
.chat-wrapper {
    max-width: 820px;
    margin: 0 auto;
}

/* ── Message bubbles ── */
.msg-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1.4rem;
    animation: fadeInUp 0.25s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user { flex-direction: row-reverse; }

.msg-avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
    margin-top: 2px;
}
.avatar-user { background: linear-gradient(135deg, #6366f1, #4f46e5); }
.avatar-ai   { background: linear-gradient(135deg, #0ea5e9, #0284c7); }

.msg-body { max-width: 78%; }

.msg-bubble {
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 14px; line-height: 1.65;
    word-wrap: break-word;
}
.bubble-user {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: #e8e9ff;
    border-bottom-right-radius: 4px;
}
.bubble-ai {
    background: #1a2035;
    border: 1px solid #1e2535;
    color: #cbd5e1;
    border-bottom-left-radius: 4px;
}
.bubble-ai p { margin: 0 0 8px 0; }
.bubble-ai p:last-child { margin-bottom: 0; }
.bubble-ai ul, .bubble-ai ol { margin: 6px 0 6px 18px; }
.bubble-ai li { margin-bottom: 4px; }
.bubble-ai strong { color: #e2e8f0; }
.bubble-ai code {
    background: #0f1117;
    border: 1px solid #1e2535;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 12.5px;
}
.bubble-ai pre {
    background: #0f1117;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 12px;
    overflow-x: auto;
}

/* ── Message metadata row ── */
.msg-meta {
    display: flex; align-items: center; gap: 8px;
    margin-top: 5px; padding: 0 4px;
}
.msg-meta.user { justify-content: flex-end; }

.agent-tag {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.4px; text-transform: uppercase;
}
.tag-rag        { background: #0c1a2e; color: #38bdf8; border: 1px solid #0369a1; }
.tag-summarizer { background: #0d1f1a; color: #34d399; border: 1px solid #065f46; }
.tag-formatter  { background: #1a1228; color: #a78bfa; border: 1px solid #5b21b6; }
.tag-conversational { background: #1a1008; color: #fbbf24; border: 1px solid #92400e; }
.tag-error      { background: #1f0c0c; color: #f87171; border: 1px solid #7f1d1d; }

.msg-time {
    font-size: 10px; color: #374151;
}

/* ── Source citation ── */
.source-bar {
    margin-top: 8px;
    padding: 7px 12px;
    background: #111827;
    border: 1px solid #1e2535;
    border-radius: 8px;
    font-size: 11px; color: #64748b;
    display: flex; align-items: flex-start; gap: 6px;
}
.source-bar-icon { color: #4b5563; flex-shrink: 0; margin-top: 1px; }

/* ── Welcome card ── */
.welcome-card {
    text-align: center;
    padding: 3rem 2rem;
    max-width: 480px;
    margin: 2rem auto;
}
.welcome-icon {
    font-size: 52px; margin-bottom: 1rem;
    filter: drop-shadow(0 0 20px rgba(99,102,241,0.5));
}
.welcome-title {
    font-size: 22px; font-weight: 700;
    color: #e2e8f0; margin-bottom: 0.5rem;
}
.welcome-desc {
    font-size: 14px; color: #64748b; line-height: 1.6;
    margin-bottom: 2rem;
}
.hint-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; text-align: left;
}
.hint-card {
    background: #1a2035;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 12px;
    cursor: pointer;
    transition: border-color 0.2s;
}
.hint-card:hover { border-color: #4f46e5; }
.hint-card-icon { font-size: 18px; margin-bottom: 5px; }
.hint-card-text { font-size: 12px; color: #94a3b8; line-height: 1.4; }

/* ── Input area ── */
.stChatInput textarea {
    background: #1a2035 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
.stChatInput textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.15) !important;
}
.stChatInput button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2535; border-radius: 4px; }

/* ── Streamlit overrides ── */
.stButton button {
    width: 100%;
    background: #1a2035;
    border: 1px solid #1e2535;
    color: #94a3b8;
    border-radius: 8px;
    font-size: 12px;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s;
}
.stButton button:hover {
    border-color: #4f46e5;
    color: #818cf8;
    background: #1e2845;
}
div[data-testid="stSpinner"] > div {
    border-top-color: #6366f1 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ── DB setup ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def setup_db():
    ingest_documents()
    return True

with st.spinner("Initializing knowledge base…"):
    st.session_state.db_ready = setup_db()

# ── Agent tag helpers ─────────────────────────────────────────────────────────
AGENT_META = {
    "rag":            ("📄", "Retrieval",    "tag-rag"),
    "summarizer":     ("📝", "Summarizer",   "tag-summarizer"),
    "formatter":      ("✉️",  "Formatter",    "tag-formatter"),
    "conversational": ("💬", "Chat",         "tag-conversational"),
    "error":          ("⚠️",  "Error",        "tag-error"),
}

def agent_tag_html(agent_type: str) -> str:
    icon, label, css = AGENT_META.get(agent_type, ("🤖", agent_type.title(), "tag-conversational"))
    return f'<span class="agent-tag {css}">{icon} {label}</span>'

def extract_source(text: str):
    """Split response text from trailing _Source: …_ line."""
    if "\n\n_Source:" in text:
        parts = text.rsplit("\n\n_Source:", 1)
        return parts[0].strip(), parts[1].strip().rstrip("_").strip()
    return text, None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🧠</div>
        <div class="sidebar-logo-text">
            <div class="sidebar-logo-title">RAG Assistant</div>
            <div class="sidebar-logo-sub">Powered by Gemini 2.5 Flash</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status
    status_label = "Knowledge base ready" if st.session_state.db_ready else "Initializing…"
    st.markdown(f'<div style="margin-bottom:1.2rem"><span class="status-pill"><span class="status-dot"></span>{status_label}</span></div>', unsafe_allow_html=True)

    # Stats
    n_msgs   = len(st.session_state.messages)
    n_user   = sum(1 for m in st.session_state.messages if m["role"] == "user")
    n_ai     = n_msgs - n_user
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-card-value">{n_user}</div>
            <div class="stat-card-label">Queries</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-value">{n_ai}</div>
            <div class="stat-card-label">Responses</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-value">{st.session_state.total_queries}</div>
            <div class="stat-card-label">Total</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Capabilities
    st.markdown('<div class="sidebar-section-title">Capabilities</div>', unsafe_allow_html=True)
    capabilities = [
        ("#38bdf8", "Document Q&A",          "Ask anything about your PDFs"),
        ("#34d399", "Smart Summarization",    "Get concise or detailed summaries"),
        ("#a78bfa", "Slack / Email Formatter","Reformat content for sharing"),
        ("#fbbf24", "Conversational AI",      "General chat & follow-ups"),
    ]
    for color, title, desc in capabilities:
        st.markdown(f"""
        <div class="cap-badge">
            <div class="cap-badge-dot" style="background:{color}"></div>
            <div>
                <div style="font-size:12px;font-weight:600;color:#cbd5e1">{title}</div>
                <div style="font-size:10px;color:#4b5563">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Actions
    st.markdown('<div class="sidebar-section-title">Actions</div>', unsafe_allow_html=True)
    if st.button("🗑️  Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div style="margin-top:0.4rem"></div>', unsafe_allow_html=True)
    if st.button("🔄  Reload knowledge base"):
        st.cache_resource.clear()
        st.session_state.db_ready = False
        st.rerun()

    # Footer
    st.markdown("""
    <div style="position:absolute;bottom:1.2rem;left:1.5rem;right:1.5rem;
                font-size:10px;color:#1f2937;text-align:center;border-top:1px solid #1e2535;
                padding-top:0.8rem">
        Multi-Agent RAG · ChromaDB · LangChain
    </div>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
col_main, col_right = st.columns([1, 0.001])   # full-width trick

with col_main:
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-header-title">🧠 RAG Agent Assistant</h1>
        <p class="main-header-sub">Intelligent document retrieval, summarization and formatting — powered by multi-agent AI.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    if not st.session_state.messages:
        # Welcome / hint screen
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">🧠</div>
            <div class="welcome-title">How can I help you today?</div>
            <div class="welcome-desc">
                Ask me anything about your uploaded documents.<br>
                I can retrieve facts, summarize, or reformat content.
            </div>
            <div class="hint-grid">
                <div class="hint-card">
                    <div class="hint-card-icon">📄</div>
                    <div class="hint-card-text">"What is the main contribution of the paper?"</div>
                </div>
                <div class="hint-card">
                    <div class="hint-card-icon">📝</div>
                    <div class="hint-card-text">"Give me a detailed summary of the documents."</div>
                </div>
                <div class="hint-card">
                    <div class="hint-card-icon">✉️</div>
                    <div class="hint-card-text">"Format the key findings as a Slack message."</div>
                </div>
                <div class="hint-card">
                    <div class="hint-card-icon">💬</div>
                    <div class="hint-card-text">"Explain XAI in simple terms."</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            role      = msg["role"]
            content   = msg["content"]
            agent_type = msg.get("agent_type", "conversational")
            ts        = msg.get("ts", "")

            if role == "user":
                st.markdown(f"""
                <div class="msg-row user">
                    <div class="msg-avatar avatar-user">👤</div>
                    <div class="msg-body">
                        <div class="msg-bubble bubble-user">{content}</div>
                        <div class="msg-meta user">
                            <span class="msg-time">{ts}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                body, source = extract_source(content)
                source_html = ""
                if source:
                    source_html = f"""
                    <div class="source-bar">
                        <span class="source-bar-icon">📎</span>
                        <span><strong style="color:#4b5563">Source:</strong> {source}</span>
                    </div>"""

                st.markdown(f"""
                <div class="msg-row ai">
                    <div class="msg-avatar avatar-ai">🤖</div>
                    <div class="msg-body">
                        <div class="msg-bubble bubble-ai">{body}</div>
                        {source_html}
                        <div class="msg-meta">
                            {agent_tag_html(agent_type)}
                            <span class="msg-time">{ts}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat input ────────────────────────────────────────────────────────────
    user_prompt = st.chat_input("Ask me anything about your documents…")

    if user_prompt:
        ts_now = time.strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user", "content": user_prompt, "ts": ts_now
        })
        st.session_state.total_queries += 1

        with st.spinner(""):
            try:
                response = run_agent(user_prompt, chat_history=st.session_state.messages)
                output     = response.get("output", "")
                agent_type = response.get("type", "conversational")
            except Exception as e:
                output     = f"An error occurred: {e}"
                agent_type = "error"

        st.session_state.messages.append({
            "role":       "assistant",
            "content":    output,
            "agent_type": agent_type,
            "ts":         time.strftime("%H:%M"),
        })
        st.rerun()
