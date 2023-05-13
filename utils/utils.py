from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
def split_chunks(sources: list) -> list:
    """
    This function splits a list of sources into chunks.

    Parameters:
    sources (list): The list of sources to be split.

    Returns:
    chunks (list): The list of chunks.
    """
    chunks = []
    splitter = RecursiveCharacterTextSplitter(separators="", chunk_size=1024, chunk_overlap=16)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

    
def generate_embedding(chunks: list, embedding, vectorstore):
    """
    This function generates embeddings for a list of chunks.

    Parameters:
    chunks (list): The list of chunks for which to generate embeddings.
    embedding (Embeddings): The embeddings to be used.
    vectorstore (VectorStore): The vector store to be used.

    Returns:
    search_index (SearchIndex): The search index that was created.
    """
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    search_index = vectorstore.from_texts(texts, embedding, metadatas=metadatas)

    return search_index


def get_files_with_extension(folder_path, extension):
    """
    This function recursively gets all the file names with a specific extension in a folder and returns them in a list.

    Parameters:
    folder_path (str): The path of the folder.
    extension (str): The extension of the files.

    Returns:
    result (list): The list of file names.
    """
    result = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                result.append(file_path)
    return result