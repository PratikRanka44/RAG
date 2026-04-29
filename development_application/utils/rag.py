# ✅ LLM

import os
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ✅ Load Data  
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/Flipping-Markets-Trading-Plan-V2.pdf")
documents = loader.load()

# ✅ Split Documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = splitter.split_documents(documents)

# ✅ Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Vector Store (FAISS)
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(texts, embeddings_model)
vectorstore.save_local("faiss_index")

# ✅ Retriever
retriever = vectorstore.as_retriever()

# ✅ Query
query = "What is Liquidity ?"

docs = retriever.invoke(query)

# ✅ Build Context
context = "\n".join([doc.page_content for doc in docs])

# ✅ Prompt
prompt = f"Answer the question in a smart way, based only on the context below:{context} Question:{query}"

# ✅ LLM Response
response = llm.invoke(prompt)

print(response.content)