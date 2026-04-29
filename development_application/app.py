import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------
# 🔧 Page Config
# -------------------------------
st.set_page_config(
    page_title="SMC Forex Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 🎨 Custom CSS — Trading Terminal Aesthetic
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap');

/* ─── Reset & Base ─── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #080B0F !important;
    color: #C9D1D9 !important;
    font-family: 'Space Mono', monospace !important;
}

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1C2333 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.2rem !important;
}

/* ─── Sidebar header ─── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #1C2333;
}
.sidebar-logo-icon {
    font-size: 1.6rem;
    background: linear-gradient(135deg, #F0B429, #FF6B35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-logo-text {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    color: #E6EDF3;
    line-height: 1.2;
}
.sidebar-logo-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #F0B429;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ─── Section labels ─── */
.sidebar-section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #484F58;
    margin: 1.4rem 0 0.6rem 0;
}

/* ─── Concept chips ─── */
.concept-chip {
    display: inline-block;
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 0.65rem;
    color: #8B949E;
    margin: 2px 2px;
    cursor: pointer;
    transition: all 0.15s ease;
    font-family: 'Space Mono', monospace;
}
.concept-chip:hover {
    border-color: #F0B429;
    color: #F0B429;
    background: #1A1E26;
}

/* ─── Stats bar ─── */
.stat-card {
    background: #0D1117;
    border: 1px solid #1C2333;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.stat-label {
    font-size: 0.55rem;
    color: #484F58;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.stat-value {
    font-size: 0.9rem;
    color: #F0B429;
    font-weight: 700;
    margin-top: 2px;
}

/* ─── Main chat area ─── */
.main-wrapper {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
}

.chat-header {
    background: #0D1117;
    border-bottom: 1px solid #1C2333;
    padding: 14px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
}

.chat-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.05rem;
    color: #E6EDF3;
    letter-spacing: 0.02em;
}

.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: #3FB950;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.status-text {
    font-size: 0.65rem;
    color: #3FB950;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ─── Chat messages container ─── */
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px 28px;
    scroll-behavior: smooth;
}

/* ─── Message bubbles ─── */
.msg-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 1.8rem;
    animation: fadeSlideIn 0.3s ease forwards;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.msg-row.user { flex-direction: row-reverse; }

.avatar {
    width: 34px;
    height: 34px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
}

.avatar.user-av {
    background: linear-gradient(135deg, #1F6FEB, #388BFD);
    color: #fff;
}

.avatar.bot-av {
    background: linear-gradient(135deg, #F0B429, #FF6B35);
    color: #080B0F;
}

.bubble {
    max-width: 72%;
    padding: 13px 18px;
    border-radius: 10px;
    font-size: 0.82rem;
    line-height: 1.75;
    letter-spacing: 0.01em;
}

.bubble.user-bubble {
    background: #1A2C4A;
    border: 1px solid #1F6FEB44;
    color: #C9D1D9;
    border-bottom-right-radius: 2px;
}

.bubble.bot-bubble {
    background: #161B22;
    border: 1px solid #1C2333;
    color: #C9D1D9;
    border-bottom-left-radius: 2px;
    position: relative;
}

.bubble.bot-bubble::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #F0B429, #FF6B35, transparent);
    border-radius: 10px 10px 0 0;
}

/* ─── Source chips ─── */
.source-row {
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}

.source-chip {
    background: #0D1117;
    border: 1px solid #21262D;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.6rem;
    color: #F0B429;
    letter-spacing: 0.08em;
    font-family: 'Space Mono', monospace;
}

/* ─── Input area ─── */
.input-wrapper {
    background: #0D1117;
    border-top: 1px solid #1C2333;
    padding: 16px 28px;
    flex-shrink: 0;
}

/* ─── Streamlit input overrides ─── */
.stTextInput > div > div {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 8px !important;
    color: #C9D1D9 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput > div > div:focus-within {
    border-color: #F0B429 !important;
    box-shadow: 0 0 0 2px #F0B42920 !important;
}
.stTextInput input {
    color: #E6EDF3 !important;
    font-family: 'Space Mono', monospace !important;
}
.stTextInput input::placeholder { color: #484F58 !important; }
.stTextInput label { display: none !important; }

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #F0B429, #FF6B35) !important;
    color: #080B0F !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    padding: 10px 22px !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px #F0B42930 !important;
    height: auto !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px #F0B42950 !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Secondary button variant */
button[kind="secondary"],
.secondary-btn > button {
    background: #161B22 !important;
    color: #8B949E !important;
    border: 1px solid #21262D !important;
    box-shadow: none !important;
}
.secondary-btn > button:hover {
    border-color: #484F58 !important;
    color: #C9D1D9 !important;
    box-shadow: none !important;
}

/* ─── Spinner ─── */
.stSpinner > div { border-top-color: #F0B429 !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #21262D; border-radius: 2px; }

/* ─── Welcome screen ─── */
.welcome-card {
    text-align: center;
    padding: 3rem 2rem;
    border: 1px solid #1C2333;
    border-radius: 12px;
    background: linear-gradient(160deg, #0D1117, #080B0F);
    margin: 3rem auto;
    max-width: 520px;
}
.welcome-icon { font-size: 2.5rem; margin-bottom: 1rem; }
.welcome-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    color: #E6EDF3;
    margin-bottom: 0.5rem;
}
.welcome-sub {
    font-size: 0.72rem;
    color: #484F58;
    line-height: 1.8;
    margin-bottom: 1.5rem;
}
.quick-label {
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #484F58;
    margin-bottom: 0.6rem;
}

/* ─── Divider ─── */
hr { border-color: #1C2333 !important; }

/* ─── Tooltip text ─── */
.tooltip-text {
    font-size: 0.62rem;
    color: #484F58;
    margin-top: 6px;
    text-align: center;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# 🔑 Load Models (cached)
# -------------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

llm = load_llm()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 📦 Session State
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # list of dicts: {role, content, sources}
if "query_input" not in st.session_state:
    st.session_state.query_input = ""


# -------------------------------
# 🔁 Handle quick-concept clicks
# -------------------------------
def set_query(text):
    st.session_state.query_input = text


# -------------------------------
# 📐 Sidebar
# -------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">📈</div>
        <div>
            <div class="sidebar-logo-text">SMC Forex</div>
            <div class="sidebar-logo-sub">Intelligence Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    msg_count = len([m for m in st.session_state.chat_history if m["role"] == "user"])
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Queries This Session</div>
        <div class="stat-value">{msg_count}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Knowledge Base</div>
        <div class="stat-value">SMC Concepts</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Concepts
    st.markdown('<div class="sidebar-section-label">Quick Concepts</div>', unsafe_allow_html=True)

    concepts = [
        ("Order Blocks", "What is an Order Block in SMC?"),
        ("BOS / CHoCH", "Explain Break of Structure and Change of Character"),
        ("Fair Value Gap", "What is a Fair Value Gap (FVG)?"),
        ("Liquidity", "What is liquidity in Smart Money Concepts?"),
        ("Inducement", "How does inducement work in SMC?"),
        ("Premium & Discount", "Explain premium and discount zones in SMC"),
        ("Market Structure", "How to identify market structure in SMC?"),
        ("Mitigation Block", "What is a mitigation block?"),
        ("NWOG / NDOG", "Explain New Week and New Day Opening Gaps"),
        ("ICT Killzones", "What are ICT killzones and when to trade them?"),
    ]

    cols = st.columns(2)
    for i, (label, query) in enumerate(concepts):
        with cols[i % 2]:
            if st.button(label, key=f"chip_{i}", use_container_width=True):
                set_query(query)

    st.markdown("---")

    # Clear chat
    st.markdown('<div class="sidebar-section-label">Session</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("🗑  Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.query_input = ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="tooltip-text" style="margin-top:1.5rem;">
        Powered by LLaMA 3.1 · FAISS · Groq
    </div>
    """, unsafe_allow_html=True)


# -------------------------------
# 💬 Main Chat Area
# -------------------------------
st.markdown("""
<div class="chat-header">
    <div class="chat-title">SMC Forex Intelligence</div>
    <div>
        <span class="status-dot"></span>
        <span class="status-text">RAG Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Messages display
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">🧠</div>
            <div class="welcome-title">Smart Money Concepts Assistant</div>
            <div class="welcome-sub">
                Ask anything about SMC — Order Blocks, BOS, FVGs,<br>
                Liquidity, Market Structure, and more.<br>
                Your knowledge base is loaded and ready.
            </div>
            <div class="quick-label">Try a concept from the sidebar →</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-row user">
                    <div class="avatar user-av">U</div>
                    <div class="bubble user-bubble">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = ""
                if msg.get("sources"):
                    chips = "".join(
                        f'<span class="source-chip">📄 Chunk {i+1}</span>'
                        for i in range(len(msg["sources"]))
                    )
                    sources_html = f'<div class="source-row">{chips}</div>'

                st.markdown(f"""
                <div class="msg-row">
                    <div class="avatar bot-av">AI</div>
                    <div class="bubble bot-bubble">
                        {msg["content"]}
                        {sources_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# -------------------------------
# 📝 Input Row
# -------------------------------
st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])

with col1:
    query = st.text_input(
        "query",
        value=st.session_state.query_input,
        placeholder="e.g. What is a Fair Value Gap and how do I trade it?",
        label_visibility="collapsed",
        key="main_input"
    )

with col2:
    send_clicked = st.button("⬆ Send", use_container_width=True)

st.markdown('<div class="tooltip-text">Press Enter or click Send · Sources cited per response</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------
# 🔍 RAG Pipeline
# -------------------------------
if (send_clicked or query) and query and query.strip():
    # Avoid re-processing the same query on rerun
    last_user_msg = st.session_state.chat_history[-1]["content"] if st.session_state.chat_history else ""
    if query.strip() != last_user_msg:
        with st.spinner("Retrieving context & reasoning..."):
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""You are an expert Smart Money Concepts (SMC) Forex analyst and educator.
Answer the question clearly and concisely using ONLY the context provided below.
Structure your answer well — use short paragraphs. If the context doesn't cover the question, say so honestly.

Context:
{context}

Question: {query}

Answer:"""

            response = llm.invoke(prompt)
            answer = response.content

            st.session_state.chat_history.append({
                "role": "user",
                "content": query.strip()
            })
            st.session_state.chat_history.append({
                "role": "bot",
                "content": answer,
                "sources": docs
            })

            # Reset input
            st.session_state.query_input = ""
            st.rerun()
