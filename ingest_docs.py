from langchain.vectorstores.faiss import FAISS
from utils.ingest import ingest_pdf
from models.embedding_model_loader import embedding_loader


if __name__ == '__main__':
    
    embeddings = embedding_loader("Cohere")
    vectorstore = FAISS
    ingest_pdf('../../data', embeddings, vectorstore, './data', 'pdf')

