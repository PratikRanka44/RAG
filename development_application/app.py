import streamlit as st
from langchain_groq import ChatGroq

# Embeddings + Vector DB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# PDF Loader + Splitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain Prompt
from langchain_core.prompts import ChatPromptTemplate

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SMC Forex RAG Assistant",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SMC Forex RAG Assistant")
st.caption(
    "Ask anything about Smart Money Concepts, Liquidity, BOS, "
    "Supply & Demand, Order Blocks, Market Structure"
)

# =========================================================
# LOAD LLM
# =========================================================

@st.cache_resource
def load_llm():

    api_key = st.secrets["GROQ_API_KEY"]

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0
    )

    return llm


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

@st.cache_resource
def create_vectorstore():

    embeddings = load_embeddings()

    # -----------------------------------------------------
    # LOAD MULTIPLE PDFs
    # -----------------------------------------------------

    pdf_files = [
        "development_application/data/file1.pdf",
        "development_application/data/file2.pdf",
        "development_application/data/file3.pdf"
    ]

    documents = []

    for pdf in pdf_files:

        loader = PyPDFLoader(pdf)

        docs = loader.load()

        documents.extend(docs)

    # -----------------------------------------------------
    # CHUNKING
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # -----------------------------------------------------
    # CREATE VECTORSTORE
    # -----------------------------------------------------

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# =========================================================
# INITIALIZE APP
# =========================================================

try:

    llm = load_llm()

    vectorstore = create_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

except Exception as e:

    st.error(f"Initialization Error: {e}")

    st.stop()


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# DISPLAY OLD CHATS
# =========================================================

for role, content in st.session_state.messages:

    with st.chat_message(role):

        st.write(content)


# =========================================================
# USER INPUT
# =========================================================

query = st.chat_input(
    "Ask about liquidity, BOS, order blocks, market structure..."
)

# =========================================================
# PROCESS QUERY
# =========================================================

if query:

    # -----------------------------------------------------
    # SHOW USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.write(query)

    st.session_state.messages.append(("user", query))

    # -----------------------------------------------------
    # RETRIEVAL + GENERATION
    # -----------------------------------------------------

    with st.spinner("Analyzing market structure..."):

        # Retrieve documents
        docs = retriever.invoke(query)

        # Combine context
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # -------------------------------------------------
        # PROMPT
        # -------------------------------------------------

        prompt = ChatPromptTemplate.from_template("""
You are an expert Smart Money Concepts (SMC) Forex mentor.

Rules:
1. Answer ONLY from the context below.
2. Keep answers clear and practical.
3. If answer is not found in context, reply EXACTLY:
NOT_FOUND

Context:
{context}

Question:
{question}
""")

        final_prompt = prompt.format_messages(
            context=context,
            question=query
        )

        # -------------------------------------------------
        # LLM RESPONSE
        # -------------------------------------------------

        response = llm.invoke(final_prompt)

        answer = response.content.strip()

        # -------------------------------------------------
        # FALLBACK RESPONSE
        # -------------------------------------------------

        if answer == "NOT_FOUND":

            answer = """
❌ We can't help with that.

💡 You can ask questions related to:

- Liquidity
- Break of Structure (BOS)
- Market Structure
- Order Blocks
- Supply & Demand
- Fair Value Gaps
- Smart Money Concepts
"""

    # -----------------------------------------------------
    # SHOW ASSISTANT RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.write(answer)

    st.session_state.messages.append(
        ("assistant", answer)
    )
