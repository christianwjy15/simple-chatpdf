import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.rag.indexing import process_pdf
from src.rag.retrieve_generation import build_graph

app = FastAPI(
    title="RAG with Memory API",
    description="An API for interacting with a conversational RAG agent."
)

graph = build_graph()
PDF_DATA_DIR = "data"


@app.post("/process-pdf")
# --- CHANGE IS HERE ---
# Renamed 'file' to 'uploaded_file' to avoid name collisions.
async def process_pdf_endpoint(uploaded_file: UploadFile = File(...)):
    """
    Endpoint to upload and process a PDF file.
    """
    if not os.path.exists(PDF_DATA_DIR):
        os.makedirs(PDF_DATA_DIR)

    # --- CHANGE IS HERE ---
    file_path = os.path.join(PDF_DATA_DIR, uploaded_file.filename)

    try:
        # Save the uploaded file temporarily
        with open(file_path, "wb") as buffer:
            # --- CHANGE IS HERE ---
            shutil.copyfileobj(uploaded_file.file, buffer)

        process_pdf(file_path)

        return {
            "status": "success",
            # --- CHANGE IS HERE ---
            "filename": uploaded_file.filename,
            "message": "File processed and indexed successfully."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# --- Chat Endpoint (No changes needed here) ---


class ChatRequest(BaseModel):
    message: str
    thread_id: str


async def stream_generator(graph_stream):
    async for step in graph_stream:
        if "generate" in step:
            generated_messages = step["generate"]["messages"]
            if generated_messages:
                ai_message = generated_messages[-1]
                if ai_message.content:
                    yield ai_message.content


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    input_data = {"messages": [("user", request.message)]}
    graph_stream = graph.astream(input_data, config=config)
    return StreamingResponse(stream_generator(graph_stream), media_type="text/plain")
