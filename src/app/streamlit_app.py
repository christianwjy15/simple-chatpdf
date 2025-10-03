import streamlit as st
import requests
import uuid
import os
from src.rag.indexing import process_pdf

# --- CONFIGURATION ---
FASTAPI_URL = "http://127.0.0.1:8000/chat"
PDF_DATA_DIR = "data"


# --- HELPER FUNCTIONS ---
def setup_directories():
    """Ensure the data directory for PDFs exists."""
    if not os.path.exists(PDF_DATA_DIR):
        os.makedirs(PDF_DATA_DIR)


# --- STREAMLIT UI ---
st.set_page_config(page_title="RAG with Memory", layout="wide")
st.title("RAG Application with Memory")


# --- SIDEBAR FOR PDF PROCESSING ---
with st.sidebar:
    st.header("Upload & Process PDF")
    uploaded_file = st.file_uploader(
        "Upload a PDF file to chat with", type="pdf"
    )

    if st.button("Process PDF"):
        setup_directories()
        file_path = os.path.join(PDF_DATA_DIR, uploaded_file.name)

        # Save the uploaded file to the data directory
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Procesing {uploaded_file.name}..."):
            try:
                process_pdf(file_path)
                st.success(
                    f"✅ Successfully processed and indexed '{uploaded_file.name}'!")
            except Exception as e:
                st.error(f"An error occured: {e}")

    else:
        st.warning("Please upload a PDF file first.")

# --- MAIN CHAT INTERFACE ---
st.header("Chat with your PDF Document")

# Initialize session state for chat history and thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your document"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in a streaming fashion
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Prepare the request payload
            payload = {"message": prompt,
                       "thread_id": st.session_state.thread_id}

            # Send request to the FastAPI backend and stream the response
            with requests.post(FASTAPI_URL, json=payload, stream=True) as r:
                r.raise_for_status()  # Raise an exception for bad status codes
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except requests.exceptions.RequestException as e:
            full_response = f"Could not connect to the backend: {e}"
            message_placeholder.error(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})
