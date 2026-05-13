# ML2 Final Practice: Local File discovery and retrieval Searching RAG

A RAG system, to allow for file discovery and content retrieval of files in the user's computer
The user can define which filetypes to include in the RAG, and query a LLM using natural language.

## What is a RAG? 🤔
Retrieval Augmented Generation (RAG) encodes documents into vector representations (embeddings) and allow to search semantically using natural language. 
A tool calling agent can query this vector storage to find relevant information, which is added to the context of the LLM, thus enriching its context with real or valuable information.

## Implementation 🧩
- The RAG is built with the [Langchain](langchain.com) python module and exposed via an [MCP Server](https://modelcontextprotocol.io/docs/develop/build-server)
- Both the LLM and embedding model are run locally using [Ollama](https://www.ollama.com/)
- [Gemini CLI](https://geminicli.com/) or  [OpenCode](https://opencode.ai/) can serve as the user interface, it is connected to the LLM (the brain) and the RAG MCP Server

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
ollama pull embeddinggemma qwen3.5:4b
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

Now you can test the rag by running
```bash
python src/ask_rag.py "find an example of matplotlib code in my code files"
```

And you will get a response like:

```md
Here are some example matplotlib code snippets found in your code files:

### 1. **Simple Plot Example** (`/home/lucky/Documents/classes/cv/code/misc/plotting.py`)
```python
import matplotlib.pyplot as plt

plt.plot(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Title")
plt.show()

### 2. **Advanced Plot Example** (`/home/lucky/Documents/classes/ml1/notes/simple_plot.py`)
```python
import matplotlib.pyplot as plt

plt.plot(data["x"], data["y1"], marker='o', linestyle='-')
plt.plot(data["x"], data["y2"], marker='o', linestyle='-')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("A plot")

ax = plt.gca()
ax[0].plot(data["x"], data["y1"], marker='o', linestyle='-')
ax[1].plot(data["x"], data["y2"], marker='o', linestyle='-')

plt.show()
```

## Gemini CLI / OpenCode Integration 🪄
For the case of Gemini CLI, simply add the following to your `~/.gemini/settings.json` file

```json
  "mcpServers": {
    "rag": {
      "command": "/your/path/to/ml2_final_practice/.venv/bin/python",
      "args": [
        "/path/to/ml2_final_practice/src/mcp_server.py"
      ],
      "trust": true
    }
  },
```

Similarly for OpenCode, add the following to `~/.config/opencode/opencode.json`:
```json
  "mcp": {
    "rag": {
      "command": [
        "/your/path/to/ml2_final_practice/.venv/bin/python",
        "/path/to/ml2_final_practice/src/mcp_server.py"
      ],
      "enabled": true,
      "type": "local"
    }
  },
```

The agents in these frontends will often try to use their own tools to answer your query, but you can ask them to "use the rag" to force them to use the RAG MCP
When they do use the rag, the results arrive much faster and are of higher quality, so the project has been proven quite useful, especially because gemini-3-flash is often terrible at using its own tools and spends ages thinking and planning.

[!Gemini example](gemini_example.png)

## Clean up 🧹
```bash
ollama rm embeddinggemma:latest qwen3.5:4b
rm -rf vector_store # or rm -rf the entire repo
```

