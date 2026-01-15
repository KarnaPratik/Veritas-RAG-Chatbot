#PDF Ingestion libraries
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

#Character splitting AI and database libraries
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))

local_path = os.path.join(current_dir, "Laptop.pdf")

if local_path:
    loader = PyPDFLoader(file_path=local_path)
    try:
        data = loader.load()
        if data:
            print(data[0].page_content)
        else:
            print("Docuement was loaded but it appears to be empty!")
    except Exception as e:
        print(f"Error loading PDF: {e}")
else:
    print("No file path provided")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True
)

chunks = text_splitter.split_documents(data)

vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = OllamaEmbeddings(model="nomic-embed-text", show_progress = True),
    # persist_directory = "./test-db", #turn on only if need to save the data for next session
    collection_name = "local-docs"
)