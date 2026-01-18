🛡️ Project Overview
Veritas AI Chatbot (Latin for Truth) addresses the "hallucination" problem in LLMs by grounding every response in user-uploaded data. Unlike standard chatbots, Veritas only answers based on provided documents and provides clickable citations for every claim.

🚀 Key Features
Secure Ingestion: PDF processing with recursive character chunking.

Semantic Search: High-dimensional vector search using ChromaDB & OLlama Embeddings.

Source Attribution: Every response includes exact citations from the source material.

Enterprise Architecture: A decoupled FastAPI (Python) and React (TypeScript) setup.

🛠️ The Tech Stack
Frontend: React, Vite, TypeScript, Tailwind CSS, Lucide Icons.

Backend: FastAPI, Python, LangChain.

Database: Pinecone (Vector DB) for long-term AI memory.

AI Model: OpenAI GPT-4o & text-embedding-3-small.
