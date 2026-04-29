
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
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 RAG Chatbot (PDF आधारित)")
st.write("Ask anything from your document")


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

    # IMPORTANT: Ensure this path exists in your GitHub repo
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
# 💬 Chat Memory
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------------
# 💬 User Input
# -------------------------------
query = st.text_input("Ask your question:")

if st.button("Ask") and query:

    with st.spinner("Thinking..."):
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a smart assistant.

Answer ONLY using the context below.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)
        answer = response.content

        # Save chat
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", answer))

        # Clear input to prevent loop
        st.rerun()


# -------------------------------
# 📜 Display Chat
# -------------------------------
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")

