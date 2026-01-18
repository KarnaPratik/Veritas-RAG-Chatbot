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
            print("Data loaded successfully")
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

#LLM interaction libraries

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever

local_model = "llama3.1"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template = """
    You are an AI assistant for Veritas AI, a document analysis system. 
    Your task is to generate 3 different search queries to find relevant information in 
    the uploaded document that would help answer the user's question.

    Generate queries that:
    - Rephrase the question using different terminology
    - Break down complex questions into simpler parts
    - Consider both technical and plain language versions

    Output only the 3 queries, one per line.

    USER QUESTION: {question}"""
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm = llm,
    prompt = QUERY_PROMPT
)

#RAG PROMPT
template = """Answer the question based ONLY on the following context:
{context}
Question : {question}
"""

prompt = ChatPromptTemplate.from_template(template)


chain = (
    {"context" : retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_question = input("Ask a question about your document: ")

response = chain.invoke(user_question)

print("\n--- AI Response---")
print(response)