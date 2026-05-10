from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma

from contextlib import contextmanager
from halo import Halo

from pathlib import Path
from os import walk, cpu_count
from os.path import join

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from math import ceil


@contextmanager
def spinner_task(text, spinner='dots'):
    spinner = Halo(text=text, spinner=spinner).start()
    try: yield spinner
    finally: spinner.succeed(text)


# https://search.brave.com/search?q=os.walk+python
def get_filepaths(starting_dir: str, suffixes: list[str], dirs_to_ignore: list[str] | None) -> list[str]:
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


def get_documents(paths: list[str]) -> list[Document]:
    pdf_paths = [p for p in paths if p.endswith('.pdf')]
    paths = list(set(paths) - set(pdf_paths))
    docs = []
    if pdf_paths: 
        for p in pdf_paths: docs.extend(PyPDFLoader(p).load())
    if docs: 
        docs.extend(UnstructuredLoader(paths).load())
    return docs
    # pdf_to_doc = lambda path: PyPDFLoader(path).load()
    # files_to_doc = lambda paths: UnstructuredLoader(paths).load()
    #
    # cpus = cpu_count() or 1
    #
    # with ProcessPoolExecutor(max_workers=cpus) as executor:
    #     results = executor.map(pdf_to_doc, pdf_paths)
    #     for doc_list in results: docs.extend(doc_list)
    #
    # # chunksize = int(ceil(len(paths) / cpus))
    # # chunks = [paths[i:i+chunksize] for i in range(0, len(paths), chunksize)]
    # chunks = [paths[i::cpus] for i in range(cpus)]
    #
    # with ProcessPoolExecutor(max_workers=cpus) as executor:
    #     results = executor.map(files_to_doc, chunks)
    #     for doc_list in results: docs.extend(doc_list)
    #
    # return docs


def generate_embeddings(model, docs, db_dir, db_table, chunk_size, chunk_overlap): 
    chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    chunks = chunker.split_documents(docs)
    vector_store = Chroma(collection_name=db_table, persist_directory=db_dir, embedding_function=model)
    chroma_max_chunks = 5461 # the maximum number of chunks you can pass to chroma.add_documents()
    for i in range(0, len(chunks), chroma_max_chunks):
        batch = chunks[i:i+chroma_max_chunks]
        vector_store.add_documents(batch)
