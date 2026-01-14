import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

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

