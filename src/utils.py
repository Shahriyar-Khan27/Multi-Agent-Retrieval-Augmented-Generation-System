import os
import hashlib
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _get_documents_fingerprint(documents_dir: str) -> str:
    """Create a fingerprint of the documents directory based on filenames, sizes, and modification times."""
    pdf_files = sorted([f for f in os.listdir(documents_dir) if f.endswith(".pdf")])
    if not pdf_files:
        return "empty"
    fingerprint_data = []
    for f in pdf_files:
        filepath = os.path.join(documents_dir, f)
        mtime = os.path.getmtime(filepath)
        size = os.path.getsize(filepath)
        fingerprint_data.append(f"{f}:{mtime}:{size}")
    return hashlib.md5("|".join(fingerprint_data).encode()).hexdigest()


def ingest_documents(db_dir: str = "chroma_db", documents_dir: str = "documents"):
    """
    Loads PDFs, splits them into chunks, and creates a ChromaDB vector store.
    Re-ingests automatically if documents have changed since last ingestion.
    """
    # Use absolute paths relative to project root
    if not os.path.isabs(db_dir):
        db_dir = os.path.join(os.getcwd(), db_dir)
    if not os.path.isabs(documents_dir):
        documents_dir = os.path.join(os.getcwd(), documents_dir)

    print(f"Checking for ChromaDB at: {db_dir}")

    current_fingerprint = _get_documents_fingerprint(documents_dir)
    fingerprint_file = os.path.join(db_dir, ".docs_fingerprint")

    # Check if DB exists and documents haven't changed
    if os.path.exists(db_dir) and os.path.exists(os.path.join(db_dir, "chroma.sqlite3")):
        if os.path.exists(fingerprint_file):
            with open(fingerprint_file, "r") as f:
                saved_fingerprint = f.read().strip()
            if saved_fingerprint == current_fingerprint:
                print("ChromaDB is up-to-date. Skipping ingestion.")
                return
            else:
                print("Documents have changed since last ingestion. Rebuilding ChromaDB...")
                shutil.rmtree(db_dir)
        else:
            # No fingerprint file means we can't verify — re-ingest to be safe
            print("No fingerprint found. Rebuilding ChromaDB to ensure consistency...")
            shutil.rmtree(db_dir)

    os.makedirs(db_dir, exist_ok=True)
    print(f"Ingesting documents from: {documents_dir}")
    print(f"Creating ChromaDB at: {db_dir}")

    documents = []
    for filename in os.listdir(documents_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(documents_dir, filename)
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                print(f"Processing: {filename}")
                doc.metadata["source"] = filename
            documents.extend(loaded_docs)

    if not documents:
        print("No documents found to ingest. Please add PDFs to the 'documents/' directory.")
        return

    # Split documents into chunks for embedding with better overlap for context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    # Use a sentence-transformer model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and persist the vector store
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_dir
    )

    # Save fingerprint so we can detect changes next time
    with open(fingerprint_file, "w") as f:
        f.write(current_fingerprint)

    print(f"Documents ingested successfully! ({len(splits)} chunks from {len(set(d.metadata.get('source') for d in documents))} files)")
