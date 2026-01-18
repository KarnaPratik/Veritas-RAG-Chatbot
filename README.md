# Veritas AI 🛡️

**Latin for "Truth"** - A Document Q&A System That Eliminates AI Hallucinations

## 🎯 Project Overview

Veritas AI addresses the "hallucination" problem in LLMs by grounding every response in user-uploaded data. Unlike standard chatbots, Veritas only answers based on provided documents, ensuring factual accuracy with source attribution.

## 🚀 Key Features

- **Secure Ingestion**: PDF processing with recursive character chunking
- **Semantic Search**: High-dimensional vector search using ChromaDB & Ollama Embeddings
- **Multi-Query Retrieval**: Generates multiple query variations for better document retrieval
- **Grounded Responses**: Answers strictly based on uploaded documents

## 🛠️ Tech Stack

**Frontend**: React, Vite, TypeScript, Tailwind CSS, Lucide Icons

**Backend**: FastAPI, Python, LangChain, LangChain Classic

**Database**: ChromaDB (Vector DB) for semantic search

**AI Models**: 
- Ollama Llama 3.1/3.2 (LLM)
- Ollama nomic-embed-text (Embeddings)

## 📋 Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed locally

## 🔧 Quick Start

### Backend Setup

```bash
# Install Ollama models
ollama pull llama3.1
ollama pull nomic-embed-text

# Install Python dependencies
cd backend
pip install fastapi uvicorn langchain langchain-community langchain-ollama langchain-classic chromadb pypdf

# Run backend
python main.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## 🏗️ How It Works

1. **Upload PDF** → Text extraction with PyPDFLoader
2. **Chunk Text** → 1000 character chunks with 200 overlap
3. **Generate Embeddings** → Ollama nomic-embed-text
4. **Store in ChromaDB** → Vector database for semantic search
5. **Multi-Query Search** → Generate 3 query variations for better retrieval
6. **Generate Answer** → Llama 3.1/3.2 responds using only document context

---

*More detailed documentation coming soon*