# 🧠 DocChat — RAG-Based Document Q&A Chatbot

> Ask questions about your own documents using AI. Upload any PDF or TXT file and get accurate, source-cited answers powered by Retrieval Augmented Generation.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://rag-chatbot.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/kavyaab2005/rag-chatbot)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge)](https://langchain.com)

---

## 📌 What is RAG?

**Retrieval Augmented Generation (RAG)** is an AI technique that:
1. **Retrieves** relevant chunks from your documents using vector similarity search
2. **Augments** the user's question with that retrieved context
3. **Generates** a grounded, accurate answer using an LLM

Unlike standard chatbots, RAG answers are **100% grounded in your documents** — no hallucinations.

---

## ✨ Features

- 📄 **Multi-format support** — Upload PDF and TXT files
- 🔍 **Semantic search** — Finds relevant content using vector embeddings
- 💬 **Natural language Q&A** — Ask questions in plain English
- 📚 **Source citations** — Shows exactly which document chunks were used
- 🧠 **Conversation memory** — Remembers previous questions in the session
- 💾 **Persistent vector store** — No need to re-index on every restart
- 🔒 **Private** — Your documents never leave your app
- 💸 **100% free stack** — Groq + Gemini free tiers

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Framework | LangChain | RAG pipeline orchestration |
| Vector Store | ChromaDB | Persistent embedding storage |
| LLM | Groq (Llama 3.1) | Answer generation |
| Embeddings | Gemini embedding-001 | Text → vector conversion |
| UI | Streamlit | Web interface |
| Deployment | Streamlit Cloud | Live hosting |

---

## 🏗️ Project Structure
rag-chatbot/

├── app.py                  # Streamlit UI

├── rag_pipeline/

│   ├── init.py

│   ├── ingest.py           # Document loading + chunking + embedding

│   ├── chain.py            # LangChain RAG chain

│   └── retriever.py

├── data/                   # Place your documents here

├── vectorstore/            # ChromaDB auto-creates this

├── .env                    # API keys (never commit!)

├── requirements.txt

└── README.md

## ⚙️ How It Works
PDF/TXT → Load → Chunk (1000 chars, 200 overlap) → Embed (Gemini) → ChromaDB

↓

User Question → Embed → Similarity Search → Top 3 Chunks → Groq LLM → Answer + Sources

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/kavyaab2005/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create a `.env` file:
GOOGLE_API_KEY=your-gemini-key-here

GROQ_API_KEY=your-groq-key-here
Get your free keys:
- Gemini: [aistudio.google.com](https://aistudio.google.com)
- Groq: [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📖 How to Use

1. **Upload** your PDF or TXT file using the sidebar
2. Click **Build Knowledge Base** — this indexes your document
3. **Ask any question** in the chat box
4. Get **accurate answers** with source citations shown below

---

## 🌐 Deployment

This app is deployed on **Streamlit Cloud** with automatic CI/CD:

- Every `git push` to `main` triggers an automatic redeploy
- API keys are stored securely as Streamlit Secrets
- Live at: [rag-chatbot.streamlit.app](https://rag-chatbot-kkega75mx9sfmx8m3oq9xt.streamlit.app/)

---

## 📁 Example Questions to Try

After uploading your resume:
What are my technical skills?

Which projects have I worked on?

Am I suitable for an AI Engineer role?

Summarize my experience in 3 points
After uploading college notes:
Explain the concept of neural networks

What are the key topics in this chapter?

Give me 5 important points from this document
## 🙋 Built By

**Kavya B**
7th Semester, CSE-AIML
Vijaya Vittala Institute of Technology, Bengaluru
VTU
---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

