# Simple Chat with PDF
[![Python 312](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

This project is a  Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents. It use FastAPI backend and a Streamlit frontend for a user-friendly, interactive experience.
<img width="1916" height="866" alt="image" src="https://github.com/user-attachments/assets/d398daf8-2bc4-4f42-aa4d-a60a11469e41" />

---

## Features
- **Interactive UI**: A clean and intuitive chat interface built with Streamlit.
- **PDF Knowledge Base**: Upload any PDF, and the application will process and index it into a vector knowledge base using ChromaDB.
- **Stateful Conversations**: The LangGraph agent maintains memory within a session, allowing for follow-up questions and contextual understanding.
- **Decoupled Architecture**: A FastAPI backend handles all AI and data processing, while the Streamlit frontend focuses solely on user interaction.

---

## Setup and Installation
Follow these steps to get the project running on your local machine.

### Prerequisites
- Python 
- uv Package Manager (if you don't have it, install with pip install uv)
- A Google AI API Key

### Installation Steps
#### 1. Clone the Repository
 ```bash
git clone https://your-repository-url.git
cd your-project-directory
```

#### 2. Create and Activate Virtual Environment with uv
```bash
# Create the virtual environment
uv venv
# Activate it (macOS/Linux)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate
```

#### 3. Install Dependencies with `uv`
This will be much faster than using traditional pip.
```bash
uv pip install -r requirements.txt
```

#### 4. Set Up Environment Variables
Create a file named `.env` in the root of the project directory and add your Google API key:
```bash
GOOGLE_API_KEY="your-google-api-key-here"
```

### Running the Application
You need to run the backend and frontend in two separate terminals.

#### 1. Start the Backend (FastAPI Server)
In your first terminal, run the following command to start the API server:

```bash
uvicorn main:app --reload
```
The server will be running at `http://127.0.0.1:8000`.

#### 2. Start the Frontend (Streamlit App)
In a second terminal, run this command to launch the user interface:

```bash
streamlit run app.py
```
Your web browser should automatically open with the application's UI.

#### 3. Using the App
Use the sidebar to upload a PDF file. Click the "Process Document" button and wait for the confirmation message.

Once processed, you can start asking questions about the document in the main chat window.

---

## Project Structure
``` bash
/simple_chatpdf
├── .venv/                            # Virtual environment managed by uv
├── chroma_db/                        # Persistent Chroma vector store
├── data/                             # Default directory for storing PDFs
├── notebooks/  
├── src/
│   ├── __init__.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexing.py               # Logic for PDF processing and component creation
│   │   └── retrieve_generation.py    # LangGraph agent definition
│   └── app/
│       ├── fastapi.py                # The FastAPI backend application
│       └── streamlit_app.py          # The Streamlit frontend application
│   
├── .env                              # Your secret API keys
├── app.py                            # The Streamlit frontend application
├── main.py                           # The FastAPI backend application
├── README.md                         # This file
└── requirements.txt                  # Project dependencies
``` 
---

## Technologies Used
- **Backend**: FastAPI, LangChain, LangGraph
- **Frontend**: Streamlit
- **LLM**: Google Gemini
- **Embeddings**: Sentence-Transformers (from Hugging Face)
- **Vector Store**: ChromaDB
- **Package Manager**: uv
---
