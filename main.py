from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
from os import getenv
from utils import spinner_task


def get

def get_pdfs(path: str):
    pdf_paths = list(Path(path).iterdir())
    docs = []
    for p in pdf_paths:
        docs.extend(PyPDFLoader(str(p)).load())
    return docs

def generate_embeddings(model, docs, db_dir, db_table, chunk_size, chunk_overlap): 
    chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    chunks = chunker.split_documents(docs)
    vector_store = Chroma(collection_name=db_table, persist_directory=db_dir, embedding_function=model)

    chroma_max_chunks = 5461 # the maximum number of chunks you can pass to chroma.add_documents()
    for i in range(0, len(chunks), chroma_max_chunks):
        batch = chunks[i:i+chroma_max_chunks]
        vector_store.add_documents(batch)


if __name__ == '__main__':
    load_dotenv()
    EMBEDDING_MODEL = getenv('EMBEDDING_MODEL')
    CHUNK_SIZE = int(getenv('CHUNK_SIZE', 0))
    CHUNK_OVERLAP = int(getenv('CHUNK_OVERLAP', 0))
    EMBEDDING_OUTPUT_DIMS = int(getenv('EMBEDDING_OUTPUT_DIMS', 0))
    EMBEDDING_DB_DIR = getenv('EMBEDDING_DB_DIR', '')
    EMBEDDING_DB_TABLE_NAME = getenv('EMBEDDING_DB_TABLE_NAME', '')
    
    # get docs -> this time all text files that are code files

    with spinner_task('Initializing embedding model'):
        model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_OUTPUT_DIMS)


    with spinner_task('Generating embeddings'):
        generate_embeddings(
            model = model,
            docs = 'IDK LOL',
            db_dir = EMBEDDING_DB_DIR,
            db_table = EMBEDDING_DB_TABLE_NAME,
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP
        )
