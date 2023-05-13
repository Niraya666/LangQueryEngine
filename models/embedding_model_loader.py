import os
from environs import load_dotenv

load_dotenv('.env')
def get_api_key(key):
    """Helper function to get API key from environment variables."""
    api_key = os.getenv(key)
    if api_key is None:
        raise ValueError(f"{key} not found. Please add it to the .env file.")
    return api_key

def embedding_loader(embeddings_name, **kwargs):
    """
    This function loads a specified embedding model.

    Parameters:
    embeddings_name (str): The name of the embedding model to be loaded. Supported models are 'openAI', 'Cohere', 'SentenceTransformerEmbeddings'.
    kwargs (dict): Additional keyword arguments to be passed to the embedding model constructor.

    Returns:
    embeddings: The loaded embedding model.

    Raises:
    ValueError: If the provided embedding model name is not supported or API key is not found in the .env file.
    """
    if embeddings_name == "openAI":
        from langchain.embeddings import OpenAIEmbeddings
        OPENAI_API_KEY = get_api_key('OPENAI_API_KEY')
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        embeddings = OpenAIEmbeddings(**kwargs)

    elif embeddings_name == "Cohere":
        from langchain.embeddings import CohereEmbeddings
        COHERE_API_KEY = get_api_key('COHERE_API_KEY')
        embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, **kwargs)
    elif embeddings_name == "SentenceTransformerEmbeddings":
        from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 

        embeddings = HuggingFaceEmbeddings(**kwargs)

    else:
        raise ValueError("Not support embedding model type!")

    return embeddings