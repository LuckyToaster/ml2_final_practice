# ML2 Final Practice: Local File discovery and retrieval Searching RAG

A RAG system, to allow for file discovery and content retrieval of files in the user's computer
The user can define which filetypes to include in the RAG, and query a LLM using natural language.

## What is a RAG? 🤔
Retrieval Augmented Generation (RAG) encodes documents into vector representations (embeddings) and allow to search semantically using natural language. 
A tool calling agent can query this vector storage to find relevant information, which is added to the context of the LLM, thus enriching its context with real or valuable information.

## Implementation 🧩
- The RAG is built with [Langchain](langchain.com) python module and exposed via an [MCP Server](https://modelcontextprotocol.io/docs/develop/build-server)
- Both the LLM and embedding model run locally using [Ollama](https://www.ollama.com/)
- [OpenCode](https://opencode.ai/) serves as the user interface, it is connected to the LLM (the brain) and the RAG MCP Server

## Requirements 📝
- `Linux environment`
- `git` 
- `python 3.14`
- `Ollama` - Install with `curl -fsSL https://ollama.com/install.sh | sh`
- `OpenCode` - Install with `curl -fsSL https://opencode.ai/install | bash`

## Launching the Project 🚀

```bash
git clone https://github.com/LuckyToaster/ml2_final_practice
cd ml2_final_practice
ollama pull embeddinggemma gemma4:e2b qwen3.5:4b qwen3-embedding:4b
python -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
```

Then, it is recommended to edit the following `.env` file's environment variables: 
- `STARTING_DIR` Where to begin scanning the filesystem, defaults to home `~` directory.
- `ALLOWED_FILETYPES` The filetypes (file suffixes) that you are interested in adding to the vector store, defaults to python files. Note that `.ipynb` extension is not currently supported.
- `DIRNAMES_TO_IGNORE` Directory names that you want to exclude from the vector store (python virtual environment directories).

```bash
python src/setup.py
```

This will scan your filesystem and find all files of interest, then run an embedding model to embed the files and store them in a vector store to be used by the RAG.


## Clean up 🧹
```bash
ollama rm embeddinggemma gemma4:e2b qwen3.5:4b qwen3-embedding:4b
rm -rf vector_store # or rm -rf the entire repo
```

