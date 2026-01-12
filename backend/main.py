import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# AI Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# 1. Load Environment Variables (API Keys)
load_dotenv()

app = FastAPI()

# 2. Safety Gate (CORS) - Allows React to talk to Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Setup AI Components
# Text Splitter: Chops PDFs into bite-sized pieces
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

# Embeddings: The "Translator" that turns text into math (vectors)
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    dimensions = 1024
)

# Pinecone: Our AI's long-term memory
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "veritas-index"

@app.get("/")
def health_check():
    return {"status": "Veritas AI is online"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    This endpoint:
    1. Receives a PDF file.
    2. Reads the text inside.
    3. Breaks it into chunks.
    4. Sends it to Pinecone.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")

    # A. Save the file temporarily so we can read it
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # B. Read the PDF
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        # C. Split into chunks
        chunks = text_splitter.split_documents(pages)

        # D. Store in Pinecone
        PineconeVectorStore.from_documents(
            chunks, 
            embeddings, 
            index_name=INDEX_NAME
        )

        return {
            "message": f"Successfully indexed {file.filename}",
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # E. Clean up: Always delete the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)