import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.rag.retrieve_generation import build_graph
from src.rag.indexing import process_pdf

# Initialize the FastAPI app
app = FastAPI(
    title="RAG with Memory API",
    description="An API for interacting with a conversational RAG"
)

# Build the LangGraph graph once when the server starts
# This is more efficient than rebuilding it for every request
graph = build_graph()


# Pydantic model for the request body to ensure type validation
class ChatRequest(BaseModel):
    message: str
    thread_id: str


# Asynchronous generator to stream graph outputs
async def stream_genarator(graph_stream):
    """
    Takes the LangGraph stream and yields the content of the AI messages.
    """

    # The graph streams dictionaries. We are interested in the final 'generate' step's output.
    async for step in graph_stream:
        # The output of a step is a dictionary where the key is the node name
        if "generate" in step:
            # The value is another dictionary, we access the 'messages'
            generated_messages = step["generate"]["messages"]
            if generated_messages:
                # The last message in the list is the new AI response
                ai_message = generated_messages[-1]
                if ai_message.content:
                    # Yield the content of the message
                    yield ai_message.content


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint to handle chat interactions. It streams the response back to the client.
    """

    # Configuration for the graph, including the thread_id for memory
    config = {"configurable": {"thread_id": request.thread_id}}

    # Prepare the input for the graph
    input_data = {"messages": [("user", request.message)]}

    # Get the asynchronous stream from the graph
    graph_stream = graph.astream(input_data, config=config)

    # Return a StreamingResponse using our generator
    return StreamingResponse(stream_genarator(graph_stream), media_type="text/plain")
