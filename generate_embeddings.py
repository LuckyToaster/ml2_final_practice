from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from os import getenv
from os.path import expanduser
from utils import spinner_task, get_filepaths, get_documents, generate_embeddings

if __name__ == '__main__':
    load_dotenv()
    EMBEDDING_MODEL = getenv('EMBEDDING_MODEL', '')
    CHUNK_SIZE = int(getenv('CHUNK_SIZE', 0))
    CHUNK_OVERLAP = int(getenv('CHUNK_OVERLAP', 0))
    EMBEDDING_OUTPUT_DIMS = int(getenv('EMBEDDING_OUTPUT_DIMS', 0))
    EMBEDDING_DB_DIR = getenv('EMBEDDING_DB_DIR', '')
    EMBEDDING_DB_TABLE_NAME = getenv('EMBEDDING_DB_TABLE_NAME', '')
    raw_suffixes = getenv('ALLOWED_FILETYPES', '')
    raw_dirnames = getenv('DIRNAMES_TO_IGNORE', '')
    SUFFIXES = [s.strip() for s in raw_suffixes.split(',') if s]
    DIRNAMES_TO_IGNORE = [s.strip() for s in raw_dirnames.split(',') if s]
    
    with spinner_task('Gathering all files'):
        starting_dir = expanduser('~')
        file_paths = get_filepaths(starting_dir, SUFFIXES, DIRNAMES_TO_IGNORE)
        print(len(file_paths))

    with spinner_task('Initializing embedding model'):
        model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_OUTPUT_DIMS)

    with spinner_task('Generating document objects from files'):
        docs = get_documents(file_paths)

    with spinner_task('Generating embeddings'):
        generate_embeddings(model, docs, EMBEDDING_DB_DIR, EMBEDDING_DB_TABLE_NAME, CHUNK_SIZE, CHUNK_OVERLAP)
