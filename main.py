from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from os import getenv
from os.path import expanduser
from utils import spinner_task, get_filepaths, get_documents, generate_embeddings

DOCUMENT_FILE_SUFFIXES = ['.txt', '.md', '.docx', '.pdf', '.ipynb']
DATA_FILE_SUFFIXES = ['.toml', '.json', '.yml' ]
CODE_FILE_SUFFIXES = ['.py', '.c', '.h', '.cpp', '.hpp', '.html', '.js', '.css']
SUFFIXES = DOCUMENT_FILE_SUFFIXES + DATA_FILE_SUFFIXES + CODE_FILE_SUFFIXES

if __name__ == '__main__':
    load_dotenv()
    EMBEDDING_MODEL = getenv('EMBEDDING_MODEL', '')
    CHUNK_SIZE = int(getenv('CHUNK_SIZE', 0))
    CHUNK_OVERLAP = int(getenv('CHUNK_OVERLAP', 0))
    EMBEDDING_OUTPUT_DIMS = int(getenv('EMBEDDING_OUTPUT_DIMS', 0))
    EMBEDDING_DB_DIR = getenv('EMBEDDING_DB_DIR', '')
    EMBEDDING_DB_TABLE_NAME = getenv('EMBEDDING_DB_TABLE_NAME', '')
    
    # get docs -> this time all text files that are code files
    # suffixes = ['.pdf', '.py']
    dirs_to_ignore = ['env', 'venv']

    with spinner_task('Gathering all files'):
        starting_dir = expanduser('~')
        file_paths = get_filepaths(starting_dir, SUFFIXES, dirs_to_ignore)

    with spinner_task('Initializing embedding model'):
        model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_OUTPUT_DIMS)

    # with spinner_task('Generating documents objects from files'):
    #     docs = get_documents(file_paths)

    # with spinner_task('Generating embeddings'):
    #     generate_embeddings(model, docs, EMBEDDING_DB_DIR, EMBEDDING_DB_TABLE_NAME, CHUNK_SIZE, CHUNK_OVERLAP)
