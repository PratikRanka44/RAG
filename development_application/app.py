
import streamlit as st
import os

# LLM
from langchain_groq import ChatGroq

# Embeddings + Vector DB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Document Loader + Splitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------
# 🔧 App Config
# -------------------------------
st.set_page_config(page_title="SMC Forex RAG Assistant", layout="wide")

st.title("📈 SMC Forex RAG Assistant")
st.caption("Ask anything about Smart Money Concepts (Liquidity, BOS,Supply & Demand, Order Blocks)")


# -------------------------------
# 🔑 Load LLM (cached)
# -------------------------------
@st.cache_resource
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("GROQ_API_KEY not found. Set it in Streamlit Secrets.")
        st.stop()

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key
    )


# -------------------------------
# 🔍 Load Embeddings (cached)
# -------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -------------------------------
# 📚 Load & Create Vectorstore
# -------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()

    # ⚠️ Ensure this path exists in your repo
    loader = PyPDFLoader("development_application/data/Flipping-Markets-Trading-Plan-V2.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore


# -------------------------------
# 🚀 Initialize Models
# -------------------------------
try:
    llm = load_llm()
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()


# -------------------------------
# 💬 Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------
# 💬 Display Chat
# -------------------------------
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)


# -------------------------------
# 💬 User Input
# -------------------------------
query = st.chat_input("Ask about liquidity, BOS, FVG, order blocks...")

if query:
    # Show user message
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Analyzing market structure..."):
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an expert Smart Money Concepts (SMC) Forex mentor.

Rules:
1. Answer ONLY from the context below.
2. If the answer is clearly NOT in the context, reply EXACTLY with: NOT_FOUND

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)
        answer = response.content.strip()

        # ✅ Fallback logic
        if answer == "NOT_FOUND":
            answer = """
❌ We can't help with that.

💡 You can ask questions related to:
- Liquidity
- Order Blocks
- Break of Structure (BOS)
- Supply & Demand
- Market Structure
"""

    # Show assistant response
    with st.chat_message("assistant"):
        st.write(answer)

    # Save messages
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("assistant", answer))

