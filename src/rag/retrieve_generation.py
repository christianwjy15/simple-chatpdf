from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from src.rag.indexing import get_llm, get_vector_store


# --- TOOL DEFINITION ---
@tool
def retrieve(query: str) -> str:
    """Retrieve relevant document snippets based on a user's query."""
    print(f"Retrieving documents for query: '{query}'")
    vector_store = get_vector_store()
    retrieved_docs = vector_store.similarity_search(query, k=3)

    context = "\n\n---\n\n".join(
        [f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}" for doc in retrieved_docs]
    )
    return context


# --- GRAPH NODES ---
def query_or_response(state: MessagesState):
    """Decide whether to call a tool for retrieval or respond directly."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}


tools = ToolNode([retrieve])


def generate(state: MessagesState):
    """
    Generate a final response using the LLM.
    - If the last message is a ToolMessage, it uses the context to answer.
    - If the last message is an AIMessage, it means the answer is already there,
      so it passes it through.
    """
    last_message = state["messages"][-1]

    # Case 1: The state contains tool output. We need to generate a response.
    if isinstance(last_message, ToolMessage):
        print("Generating response from tool output...")
        retrieved_context = last_message.content

        system_prompt = (
            "You are an expert assistant for question-answering tasks. "
            "Use the following retrieved context to answer the user's question. "
            "If you don't know the answer from the context, just say that you don't know. "
            "Keep your answer concise and to the point (max 3 sentences)."
            "\n\n## Context:\n"
            f"{retrieved_context}"
        )

        prompt_messages = [SystemMessage(
            content=system_prompt)] + state["messages"]

        llm = get_llm()
        response = llm.invoke(prompt_messages)
        return {"messages": [response]}

    # Case 2: The state already has the final AIMessage. Just return it.
    # This happens when no tool was called.
    else:
        print("Passing through direct response...")
        return {"messages": [last_message]}


# --- GRAPH BUILDER ---
def build_graph():
    """Builds and compiles the LangGraph agent."""
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("query_or_response", query_or_response)
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate", generate)

    graph_builder.set_entry_point("query_or_response")

    # If tools are called, go to "tools".
    # Otherwise (if the condition is END), go to "generate".
    graph_builder.add_conditional_edges(
        "query_or_response",
        tools_condition,
        {
            "tools": "tools",
            END: "generate"
        }
    )

    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph


# --- INTERACTIVE CHAT (for testing) ---
if __name__ == "__main__":
    graph = build_graph()
    thread_id = "user_session_123"
    config = {"configurable": {"thread_id": thread_id}}

    print("RAG Agent is ready. Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        print("Assistant: ", end="", flush=True)
        for chunk in graph.stream({"messages": [("user", user_input)]}, config=config, stream_mode="values"):
            last_message = chunk["messages"][-1]
            if last_message.content:
                print(last_message.content, end="", flush=True)
        print("\n")
