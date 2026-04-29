```python
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
        raise ValueError("GROQ_API_KEY not found. Set it in Streamlit Secrets.")

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

    # Load PDF (make sure this path exists in repo)
    loader = PyPDFLoader(
        "development_application/data/Flipping-Markets-Trading-Plan-V2.pdf"
    )
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = splitter.split_documents(documents)

    # Create FAISS index
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore


# -------------------------------
# 🚀 Initialize
# -------------------------------
llm = load_llm()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -------------------------------
# 💬 Chat Memory
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------
# 💬 Chat UI (ChatGPT style)
# -------------------------------
query = st.chat_input("Ask your question from the PDF...")

if query:
    # Show user message
    st.chat_message("user").write(query)
    st.session_state.messages.append(("user", query))

    with st.spinner("Thinking..."):
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a smart assistant.

Answer ONLY from the context below.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)
        answer = response.content

    # Show bot message
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append(("assistant", answer))


# -------------------------------
# 📜 Display Previous Chat
# -------------------------------
for role, msg in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
```
