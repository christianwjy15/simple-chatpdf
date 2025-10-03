import argparse
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_PROVIDER = "google-genai"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
VECTOR_STORE_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"

# Load environment variables
load_dotenv()


# --- COMPONENT FACTORY FUNCTIONS ---
def get_llm():
    """Returns an initialized LLM instance."""
    return init_chat_model(LLM_MODEL_NAME, model_provider=LLM_PROVIDER)


def get_embeddings():
    """Returns an initialized embeddings model instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)


def get_vector_store():
    """Returns an initialized Chroma vector store instance."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=VECTOR_STORE_PATH
    )


# --- CORE LOGIC ---
def process_pdf(file_path: str):
    """Loads, splits, and stores a PDF document in the vector store."""

    # 1. Loading documents
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    all_splits = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(all_splits)} chunks.")

    # 3. Storing documents
    vector_store = get_vector_store()
    vector_store.add_documents(documents=all_splits)
    print("PDF processed and stored in Chroma.")
    return True


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Process a PDF and store it in a Chroma vector store.")
    parser.add_argument("file_path", type=str,
                        help="The path to the PDF file to process.")
    args = parser.parse_args()
    process_pdf(args.file_path)

    # example: python src/rag/indexing.py data/indonesian-salary-survey.pdf
