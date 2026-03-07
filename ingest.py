import os

from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import config

# ------------------------------
# YouTube content loader
# ------------------------------
def load_youtube_content(url: str, save_dir: str):
    """
    Loads audio from a YouTube URL, transcribes it using FasterWhisperParser,
    and returns a list of documents.
    """
    if not url:
        print("No YouTube URL provided. Skipping YouTube ingestion.")
        return []

    print(f"Starting YouTube content loading from: {url}")
    try:
        # 1. Load audio
        loader = YoutubeAudioLoader([url], save_dir)
        # 2. Parse using FasterWhisper
        parser = FasterWhisperParser(model_size="base")  # tiny/base for low RAM
        youtube_docs = parser.load(loader)
        print(f"Successfully loaded {len(youtube_docs)} documents from YouTube.")
        return youtube_docs
    except Exception as e:
        print(f"Error loading YouTube content: {e}")
        return []

# ------------------------------
# PDF content loader
# ------------------------------
def load_pdf_content(pdf_directory: str):
    """
    Loads PDF documents from a specified directory and returns a list of pages.
    """
    print(f"Starting PDF document loading from '{pdf_directory}'...")

    if not os.path.exists(pdf_directory):
        print(f"Error: PDF directory '{pdf_directory}' not found.")
        print("Please create this directory and place your PDF files inside.")
        return []

    all_pdf_docs = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            print(f"Loading PDF document: {filepath}")
            try:
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                # Add source metadata
                for page in pages:
                    page.metadata["source"] = filename
                all_pdf_docs.extend(pages)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    if not all_pdf_docs:
        print("No PDF documents found or loaded in the specified directory.")
    else:
        print(f"Loaded {len(all_pdf_docs)} pages from PDF documents.")
    return all_pdf_docs

# ------------------------------
# Ingestion pipeline
# ------------------------------
def ingest_all_documents(
    youtube_url: str,
    youtube_save_dir: str,
    pdf_directory: str = "data",
    persist_directory: str = "docs/chroma"
):
    """
    Loads YouTube + PDF documents, splits them into chunks, generates embeddings,
    and persists them in a Chroma vector database.
    """
    print("\n--- Starting overall document ingestion process ---")

    # 1. Load YouTube content (if URL provided)
    youtube_docs = []
    if youtube_url:
        youtube_docs = load_youtube_content(youtube_url, youtube_save_dir)

    # 2. Load PDF content
    pdf_docs = load_pdf_content(pdf_directory)

    # Combine all documents
    combined_docs = youtube_docs + pdf_docs
    if not combined_docs:
        print("No documents loaded. Exiting ingestion.")
        return

    print(f"\nTotal combined documents loaded: {len(combined_docs)}")

    # 3. Split documents into chunks
    print(f"Splitting documents into chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunked_docs = splitter.split_documents(combined_docs)
    print(f"Split documents into {len(chunked_docs)} chunks.")

    # 4. Create embeddings
    print(f"Initializing embeddings with model: {config.EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    # 5. Create & persist Chroma DB
    os.makedirs(persist_directory, exist_ok=True)
    print(f"Creating Chroma vector store at '{persist_directory}'...")
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Successfully persisted Chroma DB with {len(chunked_docs)} chunks.")
    print("--- Document ingestion process complete ---")

# ------------------------------
# Run ingestion
# ------------------------------
if __name__ == "__main__":
    ingest_all_documents(
        youtube_url=config.YOUTUBE_VIDEO_URL,
        youtube_save_dir=config.YOUTUBE_AUDIO_SAVE_DIRECTORY,
        pdf_directory=config.PDF_SOURCE_DIRECTORY,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )