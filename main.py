from query_engine.query_engine import QueryEngine
from vector_store.vector_store import FAISSVectorStore
from models.llm_loader import llm_loader
from models.embedding_model_loader import embedding_loader 
"""
This script sets up a query engine to ask questions about documents embedded with a specified language model and embeddings model.

Steps:
1. Load a specific language model (in this case 'AI21') using the `llm_loader` function.
2. Load a specific embeddings model (in this case 'Cohere') using the `embedding_loader` function.
3. Initialize a FAISSVectorStore instance with the loaded embeddings model.
4. Load the vector database from a specific location ('./data/') using the `load_vector_db` method of the vector store.
5. Initialize a QueryEngine instance with the loaded language model, embeddings model, and vector store.
6. Ask a question (in this case 'what is Vector Neurons, and how is it works?') using the `ask_question_with_source` method of the query engine.
7. Print the response to the console.
"""
if __name__ == '__main__':
    # Load a specific language model
    LLM = llm_loader('AI21')

    # Load a specific embeddings model
    embeddings = embedding_loader("Cohere")

    # Initialize a FAISSVectorStore instance with the loaded embeddings model
    vector_store = FAISSVectorStore(embeddings)
    # Load the vector database from a specific location
    vector_store.load_vector_db('./data/', 'pdf')

    # Initialize a QueryEngine instance
    query_Engine = QueryEngine(LLM, embeddings, vector_store)

    # Ask a question and get the response
    result = query_Engine.ask_question_with_source('what is Vector Neurons, and how is it works?')

    print(result)
