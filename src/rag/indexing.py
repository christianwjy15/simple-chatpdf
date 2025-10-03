from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize components
# llm
LLM = init_chat_model("gemini-2.5-flash", model_provider="google-genai")

# embedding model
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")

# vector store
VECTOR_STORE = Chroma(
    collection_name="example_collection",
    embedding_function=EMBEDDINGS,
    persist_directory="./chroma_db"
)


def process_pdf(file_path: str):

    # loading documents
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    all_splits = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(all_splits)} sub-documents.")

    # storing documents
    document_ids = VECTOR_STORE.add_documents(documents=all_splits)
    print("PDF processed and stored in Chroma.")
    return True


if __name__ == "__main__":
    file_path = ("data/indonesian-salary-survey.pdf")
    process_pdf(file_path)
