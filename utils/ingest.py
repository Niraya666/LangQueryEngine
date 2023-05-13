from langchain.document_loaders import PyPDFLoader

from utils.utils import split_chunks, generate_embedding, get_files_with_extension



def ingest_pdf(files_path, embeddings, vectorstore, folder_path, index_name):
    """
    This function ingests PDF files, generates embeddings and stores them in a vector store.

    Parameters:
    files_path (str): The path where the PDF files are located.
    embeddings (Embeddings): The embeddings to be used.
    vectorstore (VectorStore): The vector store to be used.
    folder_path (str): The path where the vector store should be saved.
    index_name (str): The name of the index in the vector store.
    """
    file_path_list = get_files_with_extension(files_path, '.pdf')
    chunks = []
    for file_path in file_path_list:
        loader = PyPDFLoader(file_path)
        doc = loader.load()
        chunk = split_chunks(doc)
        chunks+=chunk
    indexs = generate_embedding(chunks, embeddings, vectorstore)

    indexs.save_local(folder_path, index_name)

    print("pdf embedding Done")