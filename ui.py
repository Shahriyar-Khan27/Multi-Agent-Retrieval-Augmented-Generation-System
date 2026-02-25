import os
import warnings

# Suppress TensorFlow / oneDNN noise — must be set before TF is imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # hide INFO + WARNING from TF C++ runtime
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"         # silence oneDNN floating-point note

# Suppress Python-level deprecation warnings from TF/Keras and PyTorch
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*tf\\..*deprecated.*")

import streamlit as st
import sys
import time
import html as html_lib
from dotenv import load_dotenv

st.set_page_config(
    page_title="RAG Agent Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agentic_rag_assistant import run_agent
from src.utils import ingest_documents

load_dotenv()

# ── SVG Icons (only used in sidebar / header / welcome — NOT inside chat) ────

_SVG_LOGO_SM = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><circle cx="4" cy="8" r="2.1" fill="white" opacity=".9"/><circle cx="4" cy="16" r="2.1" fill="white" opacity=".9"/><circle cx="12" cy="5" r="2.1" fill="white"/><circle cx="12" cy="12" r="2.1" fill="white"/><circle cx="12" cy="19" r="2.1" fill="white"/><circle cx="20" cy="9" r="2.1" fill="white" opacity=".9"/><circle cx="20" cy="15" r="2.1" fill="white" opacity=".9"/><line x1="6.1" y1="8" x2="9.9" y2="5" stroke="white" stroke-width="1" opacity=".4"/><line x1="6.1" y1="8" x2="9.9" y2="12" stroke="white" stroke-width="1" opacity=".4"/><line x1="6.1" y1="16" x2="9.9" y2="12" stroke="white" stroke-width="1" opacity=".4"/><line x1="6.1" y1="16" x2="9.9" y2="19" stroke="white" stroke-width="1" opacity=".4"/><line x1="14.1" y1="5" x2="17.9" y2="9" stroke="white" stroke-width="1" opacity=".4"/><line x1="14.1" y1="12" x2="17.9" y2="9" stroke="white" stroke-width="1" opacity=".4"/><line x1="14.1" y1="12" x2="17.9" y2="15" stroke="white" stroke-width="1" opacity=".4"/><line x1="14.1" y1="19" x2="17.9" y2="15" stroke="white" stroke-width="1" opacity=".4"/></svg>'

_SVG_LOGO_LG = '<svg width="76" height="76" viewBox="0 0 80 80" fill="none"><defs><linearGradient id="lg" x1="0" y1="0" x2="80" y2="80"><stop offset="0%" stop-color="#6366f1"/><stop offset="100%" stop-color="#8b5cf6"/></linearGradient></defs><rect width="80" height="80" rx="22" fill="url(#lg)"/><circle cx="18" cy="27" r="4.5" fill="white" opacity=".9"/><circle cx="18" cy="40" r="4.5" fill="white" opacity=".9"/><circle cx="18" cy="53" r="4.5" fill="white" opacity=".9"/><circle cx="40" cy="20" r="4.5" fill="white"/><circle cx="40" cy="40" r="4.5" fill="white"/><circle cx="40" cy="60" r="4.5" fill="white"/><circle cx="62" cy="27" r="4.5" fill="white" opacity=".9"/><circle cx="62" cy="53" r="4.5" fill="white" opacity=".9"/><line x1="22.5" y1="27" x2="35.5" y2="20" stroke="white" stroke-width="1.8" opacity=".4"/><line x1="22.5" y1="27" x2="35.5" y2="40" stroke="white" stroke-width="1.8" opacity=".4"/><line x1="22.5" y1="40" x2="35.5" y2="40" stroke="white" stroke-width="1.8" opacity=".4"/><line x1="22.5" y1="53" x2="35.5" y2="40" stroke="white" stroke-width="1.8" opacity=".4"/><line x1="22.5" y1="53" x2="35.5" y2="60" stroke="white" stroke-width="1.8" opacity=".4"/><line x1="44.5" y1="20" x2="57.5" y2="27" stroke="white" stroke-width="1.8" opacity=".4"/><line x1="44.5" y1="40" x2="57.5" y2="27" stroke="white" stroke-width="1.8" opacity=".4"/><line x1="44.5" y1="40" x2="57.5" y2="53" stroke="white" stroke-width="1.8" opacity=".4"/><line x1="44.5" y1="60" x2="57.5" y2="53" stroke="white" stroke-width="1.8" opacity=".4"/></svg>'

_SVG_FILE = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14,2 14,8 20,8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>'
_SVG_LIST = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>'
_SVG_MSG = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>'
_SVG_BULB = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a7 7 0 015 11.95V17a1 1 0 01-1 1H8a1 1 0 01-1-1v-3.05A7 7 0 0112 2z"/><line x1="9" y1="21" x2="15" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>'

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stApp { background: #0f1117; }

[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

.sidebar-logo { display:flex; align-items:center; gap:12px; padding:0.5rem 0 1.5rem 0; border-bottom:1px solid #1e2535; margin-bottom:1.5rem; }
.sidebar-logo-icon { width:42px; height:42px; background:linear-gradient(135deg,#6366f1,#8b5cf6); border-radius:12px; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.sidebar-logo-title { font-size:15px; font-weight:700; color:#e2e8f0; letter-spacing:-.3px; }
.sidebar-logo-sub { font-size:11px; color:#64748b; }
.sidebar-section-title { font-size:10px; font-weight:700; letter-spacing:1.2px; color:#4b5563; text-transform:uppercase; margin:1.4rem 0 0.6rem 0; }

.cap-badge { display:flex; align-items:center; gap:8px; padding:7px 10px; background:#1a2035; border:1px solid #1e2535; border-radius:8px; margin-bottom:6px; }
.cap-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }

.stat-row { display:flex; gap:8px; margin-bottom:1rem; }
.stat-card { flex:1; background:#1a2035; border:1px solid #1e2535; border-radius:10px; padding:10px 8px; text-align:center; }
.stat-val { font-size:20px; font-weight:700; color:#818cf8; line-height:1; }
.stat-label { font-size:10px; color:#4b5563; margin-top:3px; text-transform:uppercase; letter-spacing:.6px; }

.main-header { position:sticky; top:0; z-index:99; background:#0f1117; padding:1.2rem 0; border-bottom:1px solid #1e2535; margin-bottom:0; display:flex; align-items:center; gap:16px; }
.main-header-logo { width:46px; height:46px; background:linear-gradient(135deg,#6366f1,#8b5cf6); border-radius:14px; display:flex; align-items:center; justify-content:center; flex-shrink:0; box-shadow:0 4px 20px rgba(99,102,241,.35); }
.main-header-text h1 { font-size:24px; font-weight:700; color:#e2e8f0; letter-spacing:-.5px; margin:0 0 3px 0; }
.main-header-text p { font-size:13px; color:#64748b; margin:0; }

.status-pill { display:inline-flex; align-items:center; gap:6px; background:#0d2218; border:1px solid #134d2e; border-radius:20px; padding:4px 12px; font-size:11px; font-weight:600; color:#4ade80; }
.status-dot { width:6px; height:6px; background:#4ade80; border-radius:50%; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

.chat-wrapper { max-width:820px; margin:0 auto; padding:1rem 0; }
.msg-row { display:flex; gap:12px; margin-bottom:1.4rem; animation:fadeUp .25s ease; }
@keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
.msg-row.user { flex-direction:row-reverse; }
.msg-avatar { width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px; font-size:13px; font-weight:700; color:white; }
.av-user { background:linear-gradient(135deg,#6366f1,#4f46e5); }
.av-ai { background:linear-gradient(135deg,#0ea5e9,#0284c7); }

.msg-body { max-width:78%; }
.msg-bubble { padding:12px 16px; border-radius:16px; font-size:14px; line-height:1.65; word-wrap:break-word; }
.bubble-user { background:linear-gradient(135deg,#4f46e5,#6366f1); color:#e8e9ff; border-bottom-right-radius:4px; }
.bubble-ai { background:#1a2035; border:1px solid #1e2535; color:#cbd5e1; border-bottom-left-radius:4px; }

.msg-meta { display:flex; align-items:center; gap:8px; margin-top:6px; padding:0 4px; }
.msg-meta.user { justify-content:flex-end; }

.agent-tag { display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:20px; font-size:10px; font-weight:600; letter-spacing:.4px; text-transform:uppercase; }
.agent-tag .tag-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.tag-rag { background:#0c1a2e; color:#38bdf8; border:1px solid #0369a1; }
.tag-rag .tag-dot { background:#38bdf8; }
.tag-summarizer { background:#0d1f1a; color:#34d399; border:1px solid #065f46; }
.tag-summarizer .tag-dot { background:#34d399; }
.tag-formatter { background:#1a1228; color:#a78bfa; border:1px solid #5b21b6; }
.tag-formatter .tag-dot { background:#a78bfa; }
.tag-conv { background:#1a1008; color:#fbbf24; border:1px solid #92400e; }
.tag-conv .tag-dot { background:#fbbf24; }
.tag-error { background:#1f0c0c; color:#f87171; border:1px solid #7f1d1d; }
.tag-error .tag-dot { background:#f87171; }

.msg-time { font-size:10px; color:#374151; }

.typing-row { display:flex; gap:12px; margin-bottom:1.4rem; animation:fadeUp .25s ease; }
.typing-bubble { display:flex; align-items:center; gap:5px; padding:14px 20px; background:#1a2035; border:1px solid #1e2535; border-radius:16px; border-bottom-left-radius:4px; }
.typing-bubble span { width:8px; height:8px; border-radius:50%; background:#4b5563; animation:bounce 1.4s infinite; }
.typing-bubble span:nth-child(2) { animation-delay:.2s; }
.typing-bubble span:nth-child(3) { animation-delay:.4s; }
@keyframes bounce { 0%,60%,100%{transform:translateY(0);opacity:.4} 30%{transform:translateY(-6px);opacity:1} }

.source-bar { margin-top:8px; padding:7px 12px; background:#111827; border:1px solid #1e2535; border-radius:8px; font-size:11px; color:#64748b; display:flex; align-items:center; gap:6px; }
.source-dot { width:5px; height:5px; border-radius:50%; background:#4b5563; flex-shrink:0; }

.welcome-card { text-align:center; padding:3rem 2rem; max-width:480px; margin:2rem auto; }
.welcome-logo-wrap { display:flex; justify-content:center; margin-bottom:1.2rem; filter:drop-shadow(0 0 24px rgba(99,102,241,.5)); }
.welcome-title { font-size:22px; font-weight:700; color:#e2e8f0; margin-bottom:.5rem; }
.welcome-desc { font-size:14px; color:#64748b; line-height:1.6; margin-bottom:2rem; }

.hint-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; text-align:left; }
.hint-card { background:#1a2035; border:1px solid #1e2535; border-radius:10px; padding:12px; transition:border-color .2s; }
.hint-card:hover { border-color:#4f46e5; }
.hint-card-icon { margin-bottom:6px; }
.hint-card-text { font-size:12px; color:#94a3b8; line-height:1.4; }

.stChatInput textarea { background:#1a2035 !important; border:1px solid #1e2535 !important; border-radius:14px !important; color:#e2e8f0 !important; font-size:14px !important; font-family:'Inter',sans-serif !important; }
.stChatInput textarea:focus { border-color:#4f46e5 !important; box-shadow:0 0 0 3px rgba(79,70,229,.15) !important; }
.stChatInput button { background:linear-gradient(135deg,#4f46e5,#6366f1) !important; border-radius:10px !important; }

.stButton button { width:100%; background:#1a2035; border:1px solid #1e2535; color:#94a3b8; border-radius:8px; font-size:12px; font-family:'Inter',sans-serif; transition:all .2s; }
.stButton button:hover { border-color:#4f46e5; color:#818cf8; background:#1e2845; }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#1e2535; border-radius:4px; }
div[data-testid="stSpinner"] > div { border-top-color:#6366f1 !important; }

/* Chat container — make it transparent and fill the viewport */
section.main .block-container { padding-top:0 !important; padding-bottom:0 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { border:none !important; background:transparent !important; }
/* Welcome card vertical centering inside container */
.welcome-card { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:100%; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "processing" not in st.session_state:
    st.session_state.processing = False

# ── DB ────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def setup_db():
    ingest_documents()
    return True

with st.spinner("Initializing knowledge base…"):
    st.session_state.db_ready = setup_db()

# ── Helpers ───────────────────────────────────────────────────────────────────
AGENT_META = {
    "rag":            ("Retrieval",  "tag-rag"),
    "summarizer":     ("Summarizer", "tag-summarizer"),
    "formatter":      ("Formatter",  "tag-formatter"),
    "conversational": ("Chat",       "tag-conv"),
    "error":          ("Error",      "tag-error"),
}


def agent_tag_html(agent_type: str) -> str:
    label, css = AGENT_META.get(agent_type, (agent_type.title(), "tag-conv"))
    return f'<span class="agent-tag {css}"><span class="tag-dot"></span>{label}</span>'


def extract_source(text: str):
    if "\n\n_Source:" in text:
        parts = text.rsplit("\n\n_Source:", 1)
        return parts[0].strip(), parts[1].strip().rstrip("_").strip()
    return text, None


def render_user_msg(content: str, ts: str):
    """Render a user message — flat HTML, no indentation."""
    safe = html_lib.escape(content)
    return (
        '<div class="msg-row user">'
        '<div class="msg-avatar av-user">Y</div>'
        '<div class="msg-body">'
        f'<div class="msg-bubble bubble-user">{safe}</div>'
        f'<div class="msg-meta user"><span class="msg-time">{ts}</span></div>'
        '</div></div>'
    )


def render_ai_msg(content: str, agent_type: str, ts: str):
    """Render an AI message — flat HTML, no indentation."""
    body, source = extract_source(content)
    safe_body = body.replace("\n", "<br>")
    tag = agent_tag_html(agent_type)
    source_html = ""
    if source:
        safe_src = html_lib.escape(source)
        source_html = (
            '<div class="source-bar">'
            f'<span class="source-dot"></span>'
            f'<span><strong style="color:#4b5563">Source:</strong> {safe_src}</span>'
            '</div>'
        )
    return (
        '<div class="msg-row ai">'
        '<div class="msg-avatar av-ai">AI</div>'
        '<div class="msg-body">'
        f'<div class="msg-bubble bubble-ai">{safe_body}</div>'
        f'{source_html}'
        f'<div class="msg-meta">{tag}<span class="msg-time">{ts}</span></div>'
        '</div></div>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">'
        f'<div class="sidebar-logo-icon">{_SVG_LOGO_SM}</div>'
        '<div><div class="sidebar-logo-title">RAG Assistant</div>'
        '<div class="sidebar-logo-sub">Gemini 2.5 Flash · ChromaDB</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    status_label = "Knowledge base ready" if st.session_state.db_ready else "Initializing…"
    st.markdown(
        f'<div style="margin-bottom:1.2rem">'
        f'<span class="status-pill"><span class="status-dot"></span>{status_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")
    n_ai = len(st.session_state.messages) - n_user
    st.markdown(
        '<div class="stat-row">'
        f'<div class="stat-card"><div class="stat-val">{n_user}</div><div class="stat-label">Queries</div></div>'
        f'<div class="stat-card"><div class="stat-val">{n_ai}</div><div class="stat-label">Responses</div></div>'
        f'<div class="stat-card"><div class="stat-val">{st.session_state.total_queries}</div><div class="stat-label">Total</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">Capabilities</div>', unsafe_allow_html=True)
    for color, title, desc in [
        ("#38bdf8", "Document Q&A", "Ask anything about your PDFs"),
        ("#34d399", "Smart Summarization", "Concise or detailed summaries"),
        ("#a78bfa", "Slack / Email Formatter", "Reformat content for sharing"),
        ("#fbbf24", "Conversational AI", "General chat &amp; follow-ups"),
    ]:
        st.markdown(
            '<div class="cap-badge">'
            f'<div class="cap-dot" style="background:{color}"></div>'
            f'<div><div style="font-size:12px;font-weight:600;color:#cbd5e1">{title}</div>'
            f'<div style="font-size:10px;color:#4b5563">{desc}</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section-title">Actions</div>', unsafe_allow_html=True)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown('<div style="margin-top:.4rem"></div>', unsafe_allow_html=True)
    if st.button("Reload knowledge base"):
        st.cache_resource.clear()
        st.session_state.db_ready = False
        st.rerun()

    st.markdown(
        '<div style="position:absolute;bottom:1.2rem;left:1.5rem;right:1.5rem;'
        'font-size:10px;color:#1f2937;text-align:center;'
        'border-top:1px solid #1e2535;padding-top:.8rem">'
        'Multi-Agent RAG · LangChain · HuggingFace'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Main ──────────────────────────────────────────────────────────────────────

# 1. Header — pinned at top via CSS sticky
st.markdown(
    '<div class="main-header">'
    f'<div class="main-header-logo">{_SVG_LOGO_SM}</div>'
    '<div class="main-header-text">'
    '<h1>RAG Agent Assistant</h1>'
    '<p>Intelligent document retrieval, summarization and formatting — powered by multi-agent AI.</p>'
    '</div></div>',
    unsafe_allow_html=True,
)

# 2. Scrollable chat container — only this region scrolls
chat_container = st.container(height=550)

with chat_container:
    if not st.session_state.messages and not st.session_state.processing:
        # Welcome screen
        st.markdown(
            '<div class="welcome-card">'
            f'<div class="welcome-logo-wrap">{_SVG_LOGO_LG}</div>'
            '<div class="welcome-title">How can I help you today?</div>'
            '<div class="welcome-desc">Ask me anything about your uploaded documents.<br>'
            'I can retrieve facts, summarize, or reformat content.</div>'
            '<div class="hint-grid">'
            f'<div class="hint-card"><div class="hint-card-icon">{_SVG_FILE}</div>'
            '<div class="hint-card-text">"What is the main contribution of the paper?"</div></div>'
            f'<div class="hint-card"><div class="hint-card-icon">{_SVG_LIST}</div>'
            '<div class="hint-card-text">"Give me a detailed summary of the documents."</div></div>'
            f'<div class="hint-card"><div class="hint-card-icon">{_SVG_MSG}</div>'
            '<div class="hint-card-text">"Format the key findings as a Slack message."</div></div>'
            f'<div class="hint-card"><div class="hint-card-icon">{_SVG_BULB}</div>'
            '<div class="hint-card-text">"Explain XAI in simple terms."</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Build chat as one HTML block
        chat_html = '<div class="chat-wrapper">'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += render_user_msg(msg["content"], msg.get("ts", ""))
            else:
                chat_html += render_ai_msg(
                    msg["content"],
                    msg.get("agent_type", "conversational"),
                    msg.get("ts", ""),
                )
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    # Typing indicator placeholder (inside the scrollable container)
    typing_area = st.empty()

# 3. Process AI response — typing dots show via st.empty
if st.session_state.processing:
    typing_area.markdown(
        '<div style="max-width:820px;margin:0 auto">'
        '<div class="typing-row">'
        '<div class="msg-avatar av-ai">AI</div>'
        '<div class="typing-bubble">'
        '<span></span><span></span><span></span>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    last_user_msg = st.session_state.messages[-1]["content"]
    try:
        response = run_agent(last_user_msg, chat_history=st.session_state.messages)
        output = response.get("output", "")
        agent_type = response.get("type", "conversational")
    except Exception as e:
        output = f"An error occurred: {e}"
        agent_type = "error"

    typing_area.empty()

    st.session_state.messages.append({
        "role": "assistant",
        "content": output,
        "agent_type": agent_type,
        "ts": time.strftime("%H:%M"),
    })
    st.session_state.processing = False
    st.rerun()

# 4. Chat input — pinned at bottom by Streamlit
user_prompt = st.chat_input("Ask me anything about your documents…")

if user_prompt:
    ts_now = time.strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_prompt, "ts": ts_now})
    st.session_state.total_queries += 1
    st.session_state.processing = True
    st.rerun()
