from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from halo import Halo
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from os import walk, cpu_count, getenv
from os.path import join, expanduser
import logging
# Kill warnings from the specific library causing the spam
logging.getLogger("unstructured").setLevel(logging.ERROR)

@contextmanager
def spinner_task(text, spinner='dots'):
    spinner = Halo(text=text, spinner=spinner).start()
    try: yield spinner
    finally: spinner.succeed(text)


def get_filepaths() -> list[str]:
    raw_suffixes = getenv('ALLOWED_FILETYPES', '')
    raw_dirnames = getenv('DIRNAMES_TO_IGNORE', '')
    starting_dir = getenv('STARTING_DIR', '') or expanduser('~')

    suffixes = [s.strip() for s in raw_suffixes.split(',') if s]
    dirs_to_ignore = [s.strip() for s in raw_dirnames.split(',') if s]

    if not Path(starting_dir).is_dir():
        raise Exception('starting dir is not a directory (does not exist)')

    file_paths = []
    for root, dirs, files in walk(starting_dir): 
        dirs[:] = [d for d in dirs if not d.startswith('.')] 

        if dirs_to_ignore:
            dirs[:] = [dir for dir in dirs if not any(dir == d for d in dirs_to_ignore)] 

        for f in files:
            if any(f.endswith(s) for s in suffixes):
                file_paths.append(join(root, f))
    return file_paths


def _paths_to_doc(paths: list[str]) -> list[Document]:
    return UnstructuredLoader(paths).load()


def get_documents(paths: list[str]) -> list[Document]:
    docs = []
    cpus = cpu_count() or 1
    chunks = [paths[i::cpus] for i in range(cpus) if paths[i::cpus]] 

    with ProcessPoolExecutor(max_workers=cpus) as executor:
        results = executor.map(_paths_to_doc, chunks)
        for doc_list in results: docs.extend(doc_list) 
    return docs


def get_vector_store(embedding_model):
    return Chroma(
        collection_name=str(getenv('EMBEDDING_MODEL', '')).split(':')[0], 
        persist_directory='vector_store',
        embedding_function=embedding_model
    )


def get_text_splitter(): 
    return RecursiveCharacterTextSplitter(
        chunk_size=int(getenv('CHUNK_SIZE', 0)), 
        chunk_overlap=int(getenv('CHUNK_OVERLAP', 0)), 
        add_start_index=True
    )


def get_embedding_model():
    return  OllamaEmbeddings(
        model=getenv('EMBEDDING_MODEL', ''), 
        dimensions=int(getenv('EMBEDDING_OUTPUT_DIMS', 0))
    )


def get_llm():
    return ChatOllama(model=getenv('MODEL', ''))


def _retrieve_context(query: str, vector_store: Chroma) -> str:
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

