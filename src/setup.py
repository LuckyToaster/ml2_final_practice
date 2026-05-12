from dotenv import load_dotenv
from helpers import spinner_task, get_filepaths, get_documents, get_embedding_model, get_text_splitter, get_vector_store

if __name__ == '__main__':
    load_dotenv()

    with spinner_task('Gathering all files'):
        file_paths = get_filepaths()

    with spinner_task('Initializing embedding model'):
        model = get_embedding_model()

    with spinner_task('Generating document objects from files'):
        docs = get_documents(file_paths)

    with spinner_task(f'Embedding {len(docs)} Documents'):
        chunker = get_text_splitter()
        chunks = chunker.split_documents(docs)
        vector_store = get_vector_store(model)
        bs = 4000
        for i in range(0, len(chunks), bs):
            vector_store.add_documents(chunks[i:i+bs])
