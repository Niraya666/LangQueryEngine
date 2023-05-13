# LangQueryEngine
LangQueryEngine is a conversational AI system powered by language models, capable of ingesting PDFs, creating a searchable index, and responding to queries in a conversational manner.

## Features

1. Question and Answer dialogue using LLMs
2. Supports selection of multiple language models and embedding models
3. Efficient vector retrieval based on FAISS
4. Supports ingestion and indexing of PDF documents
5. Flexible memory options for storing and loading chat history
6. Interactive web interface based on Gradio


## Installation
```
python>=3.9
pip install -r requirements.txt
```

## Usage

### Example: Ingest PDF Files

This example demonstrates how to ingest PDF files, generate embeddings, and save them in a FAISS vector store. 

Firstly, you need to load the embeddings and initialize the vector store:

```python
from models.embedding_model_loader import embedding_loader
from langchain.vectorstores.faiss import FAISS

# Load the embeddings
embeddings = embedding_loader("Cohere")

# Initialize the vector store
vectorstore = FAISS
from utils.ingest import ingest_pdf
```
Then, you can ingest the PDF files:

```py
# Define the paths
data_path = '../../data'
save_path = './data'

# Define the index name
index_name = 'pdf'

# Ingest the PDF files
ingest_pdf(data_path, embeddings, vectorstore, save_path, index_name)

```
After running the above code, the embeddings for the ingested PDF files are saved in the specified vector store.

### Example: chat over docs


```py
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

    # Print the response
    print(result)


```

### Run webui for chat over docs

```sh
python webui.py
```
## Acknowledgments
This project was built upon the open-source Langchain project. Langchain is a modular and scalable pipeline for conversational AI, enabling the combination of different models and techniques. We extend our sincere gratitude to Langchain and its contributors.


## Roadmap
### Short-Term Goals
1. Support for more file types, not just PDFs.
2. Enhancements to the Gradio web UI for a better user experience.
3. Support for local language models and embedding models.

### Mid-Term Goals
1. Integration with additional vector stores, such as Milvus.
2. Improved mechanisms for storing and loading chat history.

## Contributing

## License


