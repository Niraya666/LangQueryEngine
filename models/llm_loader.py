import os
from environs import load_dotenv

load_dotenv('.env')

def get_api_key(key):
    """Helper function to get API key from environment variables."""
    api_key = os.getenv(key)

    if api_key is None:
        raise ValueError(f"{key} not found. Please add it to the .env file.")
    return api_key

def llm_loader(llm_name, **kwargs):
    """
    This function loads a specified Language Model (LLM).

    Parameters:
    llm_name (str): The name of the LLM to be loaded. Supported LLMs are 'openAI', 'AI21', 'GPT4All', 'Cohere'.
    kwargs (dict): Additional keyword arguments to be passed to the LLM constructor.

    Returns:
    llm: The loaded LLM.

    Raises:
    ValueError: If the provided LLM name is not supported or API key is not found in the .env file.
    """
    
    
    if llm_name == "openAI":
        from langchain.llms import OpenAI
        OPENAI_API_KEY = get_api_key('OPENAI_API_KEY')
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        llm = OpenAI(**kwargs)
    elif llm_name == "AI21":
        from langchain.llms import AI21
        AI21_API_KEY = get_api_key('AI21_API_KEY')
        llm = AI21(ai21_api_key=AI21_API_KEY, **kwargs)
    elif llm_name == "GPT4All":
        from langchain.llms import GPT4All
        llm = GPT4All(**kwargs)
    elif llm_name == "Cohere":
        from langchain.llms import Cohere
        COHERE_API_KEY = get_api_key('COHERE_API_KEY')
        llm = Cohere(cohere_api_key=COHERE_API_KEY, **kwargs)
    else:
        raise ValueError("Not support LLM model type!")
    return llm
