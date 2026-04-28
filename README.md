📄 RAG Chatbot with Groq + FAISS + Streamlit

A production-ready Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions from a PDF document. Built using **Groq LLM, **FAISS vector database, and **Streamlit UI.

🚀 Features
📄 Query PDF documents using natural language
⚡ Fast responses powered by Groq LLM
🧠 Semantic search using embeddings
💾 Persistent vector database (FAISS)
💬 Chat interface with session history
🔍 Context-aware answers (no hallucination)
🧠 How It Works
PDF → Chunking → Embeddings → FAISS Index
         ↓
User Query → Embedding → Similarity Search → Context
         ↓
Groq LLM → Answer Generation → UI Display
🛠️ Tech Stack
LLM: Groq (LLaMA 3.1)
Embeddings: HuggingFace (all-MiniLM-L6-v2)
Vector DB: FAISS
Framework: LangChain
Frontend: Streamlit
📂 Project Structure
development_application/
│
├── app.py                 # Streamlit UI
├── utils/
│   └── rag.py            # RAG backend logic
├── data/
│   └── your_pdf.pdf      # Input document
├── faiss_index/          # Saved vector database
├── .env                  # API keys (not committed)
├── requirements.txt
└── README.md
⚙️ Installation
1. Clone Repository
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
2. Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate   # Windows
3. Install Dependencies
pip install -r requirements.txt
4. Set Environment Variables

Create a .env file:

GROQ_API_KEY=your_api_key_here
▶️ Usage
Step 1: Generate Vector Store (Run Once)
python utils/rag.py

👉 This creates:

faiss_index/
Step 2: Run Streamlit App
streamlit run app.py

👉 Open in browser:

http://localhost:8501
💬 Example

User: What is liquidity?
Bot: Liquidity refers to... (based on document context)

🔒 Security Notes
❌ Do NOT hardcode API keys
✅ Always use .env or environment variables
🚀 Future Improvements
📄 Upload PDF from UI
💬 Conversational memory (multi-turn chat)
⚡ Streaming responses (ChatGPT-like UI)
📌 Show source references
🌐 Deploy on cloud (Streamlit Cloud / AWS)
🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a PR.

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Pratik Ranka

Python Developer | AI Enthusiast | Data Analyst
⭐ Support

If you found this helpful, please ⭐ the repo!

💡 Pro Tip

This project demonstrates a real-world RAG architecture used in:

AI chatbots
Document Q&A systems
Knowledge assistants