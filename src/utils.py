import os
import pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def ingest_documents(db_dir: str = "chroma_db", documents_dir: str = r"E:\Hive Pro - SHAHRIYAR\documents"):
    """
    Loads PDFs, splits them into chunks, and creates a ChromaDB vector store.
    """
    # if os.path.exists(db_dir):
    #     print("ChromaDB already exists. Skipping ingestion.")
    #     return

    documents = []
    for filename in os.listdir(documents_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(documents_dir, filename)
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                print("document is processing")
                doc.metadata["source"] = filename
            documents.extend(loaded_docs)

    if not documents:
        print("No documents found to ingest. Please add PDFs to the 'documents/' directory.")
        return

    # Split documents into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Use a sentence-transformer model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and persist the vector store
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_dir
    )
    print("Documents ingested and ChromaDB created successfully!")