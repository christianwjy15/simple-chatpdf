# app.py

import streamlit as st
import requests
import uuid
import os
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Chat with your PDF 📄",
    page_icon="🤖",
    layout="wide"
)

# Constants
CHAT_API_URL = "http://127.0.0.1:8000/chat"
PROCESS_PDF_API_URL = "http://127.0.0.1:8000/process-pdf"

# --- HELPER FUNCTIONS ---


def get_session_id():
    """Get or create a unique session ID for conversation memory."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.thread_id


# --- INITIALIZATION ---
thread_id = get_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR FOR FILE MANAGEMENT ---
with st.sidebar:
    st.header("📚 Your Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file to begin the conversation.",
        type="pdf",
        accept_multiple_files=False
    )

    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner(f"Sending '{uploaded_file.name}' for analysis..."):
                try:
                    # --- THE FIX IS ON THIS LINE ---
                    # The key 'uploaded_file' must match the parameter name in the FastAPI endpoint.
                    files = {'uploaded_file': (
                        uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                    response = requests.post(PROCESS_PDF_API_URL, files=files)

                    response.raise_for_status()

                    st.session_state.processed_pdf_name = uploaded_file.name
                    st.success(f"✅ Ready to chat with '{uploaded_file.name}'!")
                    st.session_state.messages = []

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to process document: {e}")
        else:
            st.warning("⚠️ Please upload a PDF file first.")

    if "processed_pdf_name" in st.session_state:
        st.info(
            f"**Current Document:** `{st.session_state.processed_pdf_name}`")

# --- MAIN CHAT INTERFACE (Unchanged) ---
st.title("🤖 Chat With Your PDF")
st.markdown(
    "Once you've processed a document in the sidebar, you can ask questions about it here.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if "processed_pdf_name" not in st.session_state:
        st.warning(
            "Please upload and process a PDF document before starting a conversation.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                payload = {"message": prompt, "thread_id": thread_id}

                def stream_response():
                    with requests.post(CHAT_API_URL, json=payload, stream=True) as r:
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                            yield chunk
                            time.sleep(0.01)

                full_response = st.write_stream(stream_response)

            except requests.exceptions.RequestException as e:
                error_message = f"**Connection Error:** Could not connect to the backend. (Details: {e})"
                st.error(error_message)
                full_response = error_message

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
