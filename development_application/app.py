import streamlit as st

# ✅ LLM
from langchain_groq import ChatGroq 

# ✅ Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Vector Store
from langchain_community.vectorstores import FAISS

# -------------------------------
# 🔧 Setup
# -------------------------------

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 RAG Chatbot (PDF आधारित)")
st.write("Ask anything from your document")

# -------------------------------
# 🔑 Load Models (cached)
# -------------------------------
import os
from langchain_groq import ChatGroq

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

llm = load_llm()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 💬 Chat UI
# -------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask your question:")

if st.button("Ask") and query:
    
    with st.spinner("Thinking..."):
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer smartly using ONLY the context below.

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

# -------------------------------
# 📜 Display Chat
# -------------------------------

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")