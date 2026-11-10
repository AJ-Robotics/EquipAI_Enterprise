import streamlit as st
import os
import time
import re
from datetime import datetime

# Fix for "meta tensor" error in sentence-transformers/transformers
os.environ["TRANSFORMERS_LOW_CPU_MEM_USAGE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from rag_engine import DocumentLoader, TextSplitter, VectorStore
from llm_client import LLMClient
from session_manager import SessionManager
from feedback_manager import FeedbackManager
from model_manager import ModelManager, MODEL_CATALOG
import config
import analytics

# --- Configuration ---
VECTOR_STORE_PATH = config.VECTOR_STORE_PATH
TEMP_UPLOAD_DIR = config.TEMP_UPLOAD_DIR

# --- Page Setup ---
st.set_page_config(
    page_title="EquipAI | Troubleshooting Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Shared Resources ---
@st.cache_resource
def get_vector_store():
    store = VectorStore()
    if os.path.exists(VECTOR_STORE_PATH) or os.path.exists(VECTOR_STORE_PATH + ".faiss"):
        store.load(VECTOR_STORE_PATH)
    return store

@st.cache_resource
def get_model_manager():
    return ModelManager()

@st.cache_resource
def get_llm(_mm, model_path=None):
    if not model_path:
        model_path = _mm.active_model_path()
    return LLMClient(model_path=model_path)

@st.cache_resource
def get_feedback_manager(_shared_model):
    return FeedbackManager(embedding_model=_shared_model)

# --- Session Initialization ---
if "store" not in st.session_state:
    st.session_state.store = get_vector_store()
if "model_manager" not in st.session_state:
    st.session_state.model_manager = get_model_manager()
if "llm" not in st.session_state:
    _active = st.session_state.model_manager.active_model_path()
    st.session_state.llm = get_llm(st.session_state.model_manager, _active)
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()
if "feedback_manager" not in st.session_state:
    st.session_state.feedback_manager = get_feedback_manager(st.session_state.store.model)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = st.session_state.session_manager.create_new_session()
    st.session_state.session_title = "New Chat"
if "theme" not in st.session_state:
    st.session_state.theme = "Classic"
if "response_mode" not in st.session_state:
    st.session_state.response_mode = config.DEFAULT_RESPONSE_MODE

# --- CSS Overhaul (Classic Premium) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #ffffff;
        --surface: #f8fafc;
        --text: #0f172a;
        --accent: #2563eb;
        --border: #e2e8f0;
    }

    .stApp { background-color: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
    
    /* Branding */
    .brand { font-weight: 700; font-size: 22px; color: var(--text); margin-bottom: 20px; }
    .brand span { color: var(--accent); }

    /* Custom Status */
    .status-box { 
        display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; 
        background: var(--surface); border: 1px solid var(--border); border-radius: 8px; 
        font-size: 12px; font-weight: 500; color: var(--text); margin-bottom: 24px;
    }
    .dot-ready { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 8px rgba(34, 197, 94, 0.4); }

    /* Sidebar Labels */
    .sidebar-label { 
        font-size: 10px; color: #64748b; text-transform: uppercase; 
        letter-spacing: 1px; margin: 24px 0 8px; display: block;
        border-bottom: 1px solid var(--border); padding-bottom: 4px;
    }

    /* Production Chat Enhancements */
    [data-testid="stChatMessage"] { 
        padding: 1.5rem !important; border-radius: 12px !important; 
        border: 1px solid var(--border) !important; margin-bottom: 1rem !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="user-avatar"]) {
        background-color: #f1f5f9 !important;
        border: none !important;
    }

    .fault-tag { background: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; }
    .citation-tag { font-size: 11px; color: #64748b; margin-top: 10px; font-style: italic; display: block; }
    
    /* Clean Hide */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Logic Helpers ---
def handle_feedback(msg_idx, rating):
    msg = st.session_state.messages[msg_idx]
    if msg_idx > 0 and st.session_state.messages[msg_idx-1]['role'] == 'user':
        user_query = st.session_state.messages[msg_idx-1]['content']
        bot_response = msg['content']
        context_used = msg.get('citations', [])
        st.session_state.feedback_manager.add_feedback(user_query, bot_response, rating, context_used)
        st.toast("✅ Feedback recorded.")

def new_chat():
    if st.session_state.messages:
        st.session_state.session_manager.save_session(
            st.session_state.current_session_id, 
            st.session_state.messages,
            st.session_state.session_title
        )
    st.session_state.current_session_id = st.session_state.session_manager.create_new_session()
    st.session_state.messages = []
    st.session_state.session_title = "New Chat"
    st.rerun()

def load_chat(session_id):
    data = st.session_state.session_manager.load_session(session_id)
    if data:
        st.session_state.current_session_id = data["session_id"]
        st.session_state.messages = data["messages"]
        st.session_state.session_title = data.get("title", "Saved Session")
    st.rerun()

def delete_chat(session_id):
    st.session_state.session_manager.delete_session(session_id)
    if st.session_state.current_session_id == session_id: new_chat()
    else: st.rerun()

def export_conversation():
    """Exports the current conversation to a Markdown file for reports."""
    if not st.session_state.messages:
        st.toast("No messages to export yet.")
        return
    lines = [f"# EquipAI Troubleshooting Report", ""]
    lines.append(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Session:** {st.session_state.session_title}")
    lines.append(f"- **Role:** {st.session_state.get('role','Technician')}")
    lines.append(f"- **Machine:** {st.session_state.get('sel_machine','All')}")
    lines.append(f"- **Subsystem:** {st.session_state.get('sel_subsystem','All')}")
    lines.append("")
    for msg in st.session_state.messages:
        name = "**Operator**" if msg["role"] == "user" else "**EquipAI**"
        lines.append(f"{name}: {msg['content']}")
        for c in msg.get("citations", [])[:2]:
            lines.append(f"   > 📄 *Reference: {os.path.basename(c.get('source','Manual'))}*")
        lines.append("")
    md = "\n".join(lines)
    fname = f"equipai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    st.download_button("⬇️ Download Report (.md)", md, file_name=fname,
                       mime="text/markdown", use_container_width=True)
    st.toast("Report ready for download.")

def apply_model(model_path):
    """Activates a model and reloads the engine."""
    if st.session_state.model_manager.set_active(model_path):
        get_llm.clear()
        st.session_state.llm = get_llm(st.session_state.model_manager, model_path)
        st.toast(f"✅ Model switched to {os.path.basename(model_path)}")
        st.rerun()
    else:
        st.error("Could not activate that model.")

def download_catalog_model(catalog_key):
    """Downloads a catalog model with progress."""
    entry = next((m for m in MODEL_CATALOG if m["key"] == catalog_key), None)
    if not entry:
        return
    with st.status(f"⬇️ Downloading {entry['name']} ({entry['size_gb']} GB)...", expanded=True) as status:
        prog = st.progress(0.0)
        def _hook(fraction, label):
            prog.progress(fraction)
        try:
            path = st.session_state.model_manager.download(catalog_key, progress=_hook)
            status.update(label=f"✅ Download complete: {os.path.basename(path)}", state="complete")
            st.toast("Download complete. Switch to this model in Model Management.")
            st.rerun()
        except Exception as e:
            status.update(label=f"❌ Download failed: {e}", state="error")

def ingestion_workflow(uploaded_files, machine, subsystem):
    if not uploaded_files: return
    with st.status("🧠 Updating Knowledge Base...", expanded=True) as status:
        loader = DocumentLoader()
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        docs = []
        if not os.path.exists(TEMP_UPLOAD_DIR): os.makedirs(TEMP_UPLOAD_DIR)
        for f in uploaded_files:
            p = os.path.join(TEMP_UPLOAD_DIR, f.name)
            with open(p, "wb") as out: out.write(f.getbuffer())
            d = loader.load_file(p)
            if d:
                d['machine'], d['subsystem'] = machine, subsystem
                docs.append(d)
        if docs:
            st.session_state.store.add_documents(docs, splitter)
            st.session_state.store.save(VECTOR_STORE_PATH)
            status.update(label="✅ Knowledge Base Synchronized", state="complete")
            time.sleep(1); st.rerun()
        else: status.update(label="❌ Text extraction failed", state="error")

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="brand">Equip<span>AI</span> Enterprise</div>', unsafe_allow_html=True)
    is_ready = st.session_state.llm.check_health()
    st.markdown(f'<div class="status-box"><div class="dot-ready" style="background:{"#22c55e" if is_ready else "#ef4444"}"></div>{"System Active" if is_ready else "Check Backend"}</div>', unsafe_allow_html=True)
    
    if st.button("✦ New Session", type="primary", use_container_width=True): new_chat()
    
    st.markdown('<span class="sidebar-label">Context Management</span>', unsafe_allow_html=True)
    user_role = st.selectbox("Your Role", ["Operator", "Technician", "Engineer"], index=1)
    st.session_state.role = user_role

    st.markdown('<span class="sidebar-label">Response Mode</span>', unsafe_allow_html=True)
    _mode_map = {f"{m['icon']} {m['label']}": k for k, m in config.RESPONSE_MODES.items()}
    _mode_options = list(_mode_map.keys())
    _current_mode = config.RESPONSE_MODES.get(st.session_state.response_mode, config.RESPONSE_MODES["balanced"])
    _sel_mode_label = f"{_current_mode['icon']} {_current_mode['label']}"
    response_mode = st.radio(
        "Response Mode",
        options=_mode_options,
        index=_mode_options.index(_sel_mode_label),
        label_visibility="collapsed",
        help=config.RESPONSE_MODES[st.session_state.response_mode]["usage"],
    )
    st.session_state.response_mode = _mode_map[response_mode]
    st.caption(config.RESPONSE_MODES[st.session_state.response_mode]["usage"])

    with st.expander("🧠 Model Management"):
        mm = st.session_state.model_manager
        installed = mm.list_main_models()
        active = mm.active_model_path()
        st.caption(f"**Active:** {os.path.basename(active) if active else 'None'}")
        if not installed:
            st.caption("No local models found. Download one below.")
        for m in installed:
            is_active = m["path"] == active
            st.markdown(
                f"✓ **{m['name']}** · {m['size']}" if is_active else
                f"• {m['name']} · {m['size']}",
                help=m["path"])
            if not is_active:
                c1, c2 = st.columns([0.6, 0.4])
                if c1.button("Use", key=f"use_{m['path']}",
                             use_container_width=True):
                    apply_model(m["path"])
                if c2.button("Delete", key=f"del_{m['path']}",
                             use_container_width=True):
                    try:
                        mm.delete_model(m["path"])
                        st.toast(f"🗑️ Deleted {os.path.basename(m['path'])}")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        st.divider()
        st.caption("Download from catalog")
        catalog_names = {m["key"]: f"{m['name']} ({m['size_gb']} GB)" for m in MODEL_CATALOG}
        sel_cat = st.selectbox("Model", options=list(catalog_names.keys()),
                               format_func=lambda k: catalog_names[k],
                               label_visibility="collapsed")
        if st.button(f"⬇️ Download selected", use_container_width=True):
            download_catalog_model(sel_cat)
    
    all_metas = list(st.session_state.store.metadata.values()) if type(st.session_state.store.metadata) == dict else st.session_state.store.metadata
    machines = sorted(list(set([m.get("machine", "CNC-01") for m in all_metas if m.get("machine")] + ["CNC-01"])))
    subsystems = sorted(list(set([m.get("subsystem", "Mechanical") for m in all_metas if m.get("subsystem")] + ["General"])))
    
    sel_machine = st.selectbox("Machine Unit", ["All"] + machines)
    sel_subsystem = st.selectbox("Subsystem", ["All"] + subsystems)
    st.session_state.sel_machine = sel_machine
    st.session_state.sel_subsystem = sel_subsystem

    with st.expander("⚙️ Advanced Retrieval"):
        use_hybrid = st.toggle("Hybrid search (code + vector)", config.USE_HYBRID)
        ret_threshold = st.slider("Similarity threshold", 0.0, 0.5,
                                  config.RETRIEVAL_THRESHOLD, 0.05)

    with st.expander("📊 System Telemetry"):
        for k, v in st.session_state.llm.model_info.items():
            st.caption(f"{k}: {v}")
        info = st.session_state.store.get_stats()
        st.caption(f"Knowledge chunks: {info['chunks']}")
        st.caption(f"Documents: {info['documents']}")
        st.caption(f"Fault codes indexed: {info['fault_codes_indexed']}")
        st.caption(f"Device: {info['device']}")
        st.button("Export Analytics (.csv)",
                  on_click=lambda: st.toast(analytics.export_csv(
                      f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")), 
                  use_container_width=True)

    with st.expander("📤 Export Conversation"):
        export_conversation()
    
    st.markdown('<span class="sidebar-label">Session History</span>', unsafe_allow_html=True)
    sessions = st.session_state.session_manager.list_sessions()
    if not sessions: st.caption("No recent history")
    for s in sessions[:8]:
        c1, c2 = st.columns([0.8, 0.2])
        with c1:
            if st.button(f"🔧 {s['title'][:20]}", key=f"s_{s['session_id']}", use_container_width=True): load_chat(s['session_id'])
        with c2:
            if st.button("✕", key=f"d_{s['session_id']}"): delete_chat(s['session_id'])

    st.markdown('<span class="sidebar-label">Administration</span>', unsafe_allow_html=True)
    with st.expander("Knowledge Import"):
        m_tag = st.text_input("Machine ID", "CNC-01")
        s_tag = st.selectbox("System Type", ["Electrical", "Mechanical", "Hydraulic", "Software"])
        files = st.file_uploader("Upload Manuals", accept_multiple_files=True)
        if st.button("Synchronize Docs", use_container_width=True): ingestion_workflow(files, m_tag, s_tag)

# --- Main Interface ---
if not st.session_state.llm.check_health():
    st.error("⚠️ AI Backend Offline. Please check model path."); st.stop()

# Header
mode_icon = config.RESPONSE_MODES[st.session_state.response_mode]["icon"]
model_name = os.path.basename(st.session_state.llm.model_info["name"])
device = st.session_state.store.get_stats()["device"]
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:4px;">
        <h2 style="margin:0; font-weight:800; letter-spacing:-0.5px;">
            {sel_machine if sel_machine != 'All' else 'Global'} Troubleshooting
        </h2>
        <span style="background:#f1f5f9; border:1px solid #e2e8f0; border-radius:8px; padding:3px 10px; font-size:12px; color:#334155;">
            👤 {user_role} mode
        </span>
        <span style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:3px 10px; font-size:12px; color:#1d4ed8;">
            {mode_icon} {config.RESPONSE_MODES[st.session_state.response_mode]['label']} response
        </span>
        <span style="background:#ecfdf5; border:1px solid #a7f3d0; border-radius:8px; padding:3px 10px; font-size:12px; color:#047857;">
            🟢 {model_name} · {device}
        </span>
    </div>
    """, unsafe_allow_html=True)
st.caption(f"Session Focus: {sel_subsystem if sel_subsystem != 'All' else 'Full System'}")

if not st.session_state.messages:
    st.markdown(f"""
        <div style="padding: 100px 0; text-align: center;">
            <h1 style="font-weight: 800; letter-spacing: -1px;">EquipAI ready.</h1>
            <p style="color: #64748b; font-size: 18px;">How can I assist with your machinery maintenance today, {user_role}?</p>
        </div>
    """, unsafe_allow_html=True)

# Render Chat History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Highlight fault codes in content
        content = re.sub(r'([A-Z][0-9]{3}|[S|E][0-9]{3})', r'<span class="fault-tag">\1</span>', msg["content"])
        st.markdown(content, unsafe_allow_html=True)
        
        if msg["role"] == "assistant":
            citations = "".join([f'<span class="citation-tag">📄 Reference: {os.path.basename(c.get("source", "Manual"))}</span>' for c in msg.get("citations", [])[:2]])
            if citations: st.markdown(citations, unsafe_allow_html=True)
            
            c1, c2, _ = st.columns([0.05, 0.05, 0.9])
            with c1: st.button("👍", key=f"up_{idx}", on_click=handle_feedback, args=(idx, 1))
            with c2: st.button("👎", key=f"down_{idx}", on_click=handle_feedback, args=(idx, -1))

# Chat Input & Logic
if prompt := st.chat_input("Input fault code or description..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%I:%M %p")})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        # Retrieval (mode-aware)
        filters = {}
        if sel_machine != "All": filters['machine'] = sel_machine
        if sel_subsystem != "All": filters['subsystem'] = sel_subsystem

        mode_cfg = config.response_mode(st.session_state.response_mode)
        t0 = analytics.timer_start()
        results = st.session_state.store.search(
            user_query, k=mode_cfg["k"], filter_criteria=filters,
            threshold=ret_threshold, hybrid=use_hybrid,
        )
        
        # Display response with streaming
        response_placeholder = st.empty()
        full_response = ""

        for chunk in st.session_state.llm.generate_answer_stream(
            user_query, results, role=user_role,
            mode=st.session_state.response_mode):
            full_response += chunk
            # Preview formatting
            display_text = re.sub(r'([A-Z][0-9]{3}|[S|E][0-9]{3})', r'<span class="fault-tag">\1</span>', full_response)
            response_placeholder.markdown(display_text, unsafe_allow_html=True)

        latency = analytics.timer_ms(t0)
        code_match = any(r.get("code_match") for r in results[:3])
        analytics.log_query(
            user_query, role=user_role, machine=sel_machine, subsystem=sel_subsystem,
            latency_ms=latency,
            retrieval_score=(results[0]["score"] if results else None),
            sources=[r["metadata"].get("source") for r in results[:3]],
            code_match=code_match, session_id=st.session_state.current_session_id,
            mode=st.session_state.response_mode,
        )

        # Store AI message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "citations": [r['metadata'] for r in results]
        })
        
        # Update title if new session
        if len(st.session_state.messages) <= 2:
            st.session_state.session_title = user_query[:30] + "..."
            
        st.session_state.session_manager.save_session(
            st.session_state.current_session_id, 
            st.session_state.messages,
            st.session_state.session_title
        )
        st.rerun()

# Global Footer
st.markdown("<div style='text-align:center; padding: 60px 0; color:#94a3b8; font-size:11px;'>EquipAI Enterprise TROUBLESHOOTING | VERSION 1.0.0 | jocichanjosepheee@gmail.com</div>", unsafe_allow_html=True)
