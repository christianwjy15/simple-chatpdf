from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from src.rag.indexing import get_llm, get_embeddings, get_vector_store


# --- TOOL DEFINITION ---
@tool
def retrieve(query: str) -> str:
    """Retrieve relevant document snippets based on a user's query."""
    print(f"Retrieving documents for query: '{query}'")
    vector_store = get_vector_store()
    retrieved_docs = vector_store.similarity_search(query, k=3)

    # Format the retrieved documents into a single string
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


# Set up the ToolNode with our retrieve tool
tools = ToolNode([retrieve])


def generate(state: MessagesState):
    """Generate a final response using the LLM and retrieved context."""
    # The tool output is always the last message
    last_message = state["messages"][-1]

    # Ensure the last message is a ToolMessage before proceeding
    if not isinstance(last_message, ToolMessage):
        return {"messages": [SystemMessage(content="Error: Expected tool output.")]}

    # The content of the ToolMessage is the context from our `retrieve` tool
    retrieved_context = last_message.content

    system_prompt = (
        "You are an assistant for question answering task. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n## Context:\n"
        f"{retrieved_context}"
    )

    # Prepend the system prompt to the conversation history
    prompt_messages = [SystemMessage(
        content=system_prompt)] + state["messages"]

    llm = get_llm()
    response = llm.invoke(prompt_messages)
    return {"messages": [response]}


# --- GRAPH BUILDER ---
def build_graph():
    """Builds and compiles the LangGraph agent."""
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("query_or_response", query_or_response)
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate", generate)

    graph_builder.set_entry_point("query_or_response")

    # If the LLM calls a tool, go to the 'tools' node. Otherwise, end.
    graph_builder.add_conditional_edges(
        "query_or_response",
        tools_condition,
        {END: END, "tools": "tools"}
    )

    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph


if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "aaa111"}}

    # --- INTERACTIVE CHAT ---
    print("RAG Agent is ready. Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Stream the response
        print("Assistant: ", end="", flush=True)
        for chunk in graph.stream({"messages":
                                   [("user", user_input)]},
                                  config=config,
                                  stream_mode="values"):
            # The last message is the new one
            last_message = chunk["messages"][-1]
            if last_message.content:
                print(last_message.content, end="", flush=True)
        print("\n")
