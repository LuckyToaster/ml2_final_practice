# ML2 Final Practice: Code Searching RAG

A RAG system, using [OpenCode](https://opencode.ai/) and [Ollama](https://www.ollama.com/) to find code snippets and files on your machine, using natural language.

## What is a RAG?
Retrieval Augmented Generation (RAG) encodes documents into vector representations (embeddings) and allow to search semantically using natural language. 
A tool calling agent can query this vector storage to find relevant information, which is added to the context of the LLM, therefore enriching it with real or valuable information.

## Implementation
- The RAG is built with [Langchain](langchain.com) python module and exposed via an [MCP Server](https://modelcontextprotocol.io/docs/develop/build-server)
- Both the LLM and embedding model run locally using [Ollama](https://www.ollama.com/)
- [OpenCode](https://opencode.ai/) serves as the user interface, it is connected to the LLM (the brain) and the RAG MCP Server

## Requirements
- `Linux environment`
- `git` 
- `python 3.14`
- `Ollama` - Install with `curl -fsSL https://ollama.com/install.sh | sh`
- `OpenCode` - Install with `curl -fsSL https://opencode.ai/install | bash`


## Launching the Project
```bash
git clone https://github.com/LuckyToaster/ml2_final_practice
cd ml2_final_practice
ollama pull embeddinggemma gemma4:e2b
python -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
```
