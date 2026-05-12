from fastmcp import FastMCP
from dotenv import load_dotenv
from helpers import get_vector_store, get_embedding_model


load_dotenv()
vector_store = get_vector_store(get_embedding_model())
mcp = FastMCP('RAG')


@mcp.tool
def retrieve_context(query: str) -> str:
    """Search codebase. Returns file paths and relevant code blocks."""
    retrieved_docs = vector_store.similarity_search(query, k=8)
    if not retrieved_docs: return "Error: No files matching that query were found."

    files = {}
    for doc in retrieved_docs:
        path = doc.metadata.get('source', 'unknown_file')
        if path not in files: files[path] = []
        files[path].append(doc.page_content)

    output = []
    for path, chunks in files.items():
        formatted_file = f"FILE_PATH: {path}\n"
        formatted_file += "--- RELEVANT SNIPPETS ---\n"
        formatted_file += "\n[...]\n".join(chunks)
        formatted_file += "\n--- END OF FILE ---"
        output.append(formatted_file)

    return "\n\n================================\n\n".join(output)

# @mcp.tool
# def retrieve_context(query: str) -> str:
#     """Retrieve information from the knowledge base to help answer a query."""
#     retrieved_docs = vector_store.similarity_search(query, k=10)
#     if not retrieved_docs: return "No relevant information found."
#
#     serialized = "\n\n".join(
#         f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
#         for doc in retrieved_docs
#     )
#     return serialized

@mcp.prompt()
def rag_assistant(query: str):
    return f"""You are a Code Navigation Agent. Your primary goal is to map the user's request to specific files and code implementation details.

    USER QUERY: "{query}"

    OPERATING PROTOCOL:
    1. **Identify Entry Points**: Use 'retrieve_context' to find where the logic starts (e.g., API routes, main functions).
    2. **Follow the Trace**: If the retrieved code references a class or function in another file, immediately mention you need to search for that specific symbol.
    3. **Precision over Prose**: Do not summarize code; locate it. Provide the exact file path and the line/snippet where the logic resides.
    4. **Output Structure**:
       - **Location**: [File Path]
       - **Definition**: [Function/Class name]
       - **Implementation**: [Snippet]

    If you cannot find the exact file, list the files that are 'closest' based on naming conventions found in the context.

    EXECUTE SEARCH."""


if __name__ == '__main__':
    mcp.run()
