from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    """
    This is an abstract base class for a vector store.
    """
    def __init__(self, embeddings):
        self.embeddings = embeddings
    @abstractmethod   
    def insert_index(self):
        """
        An abstract method for inserting indexes into the vector store.
        """
        pass
    @abstractmethod
    def load_vector_db(self,**kwargs):
        """
        An abstract method for loading the vector database.
        """
        pass
    @abstractmethod
    def save_vector_db(self, **kwargs):
        """
        An abstract method for saving the vector database.
        """
        pass
    @abstractmethod
    def search(self, **kwargs):
        """
        An abstract method for searching in the vector store.
        """
        pass

class FAISSVectorStore(BaseVectorStore):
    """
    This is a class for a FAISS vector store.
    """
    def __init__(self,embeddings):
        super(FAISSVectorStore, self).__init__(
            embeddings = embeddings
        )
        from langchain.vectorstores.faiss import FAISS
        self.vectorstore = FAISS

    def create_index(self,
                     texts:List[str],
                     metadatas:Optional[List[dict]] = None):
        """
        This method creates an index for the given texts and metadata.
        """
        self.vector_db = self.vectorstore.from_texts(texts, self.embeddings, metadatas=metadatas)
        
    def insert_index(self, texts, metadatas):
        """
        This method inserts an index for the given texts and metadata into the vector store.
        """
        self.vector_db.add_texts(texts, metadatas=metadatas)
        
    def save_vector_db(self,folder_path: str, index_name: str = "index"):
        """
        This method saves the vector database to a local file.
        """
        self.vector_db.save_local(folder_path, index_name)
        
    def load_vector_db(self, folder_path: str, index_name: str = "index"):
        """
        This method for loading the vector database from a local file.
        """
        self.vector_db = self.vectorstore.load_local(folder_path,self.embeddings, index_name)
        
    def search(self,query:str, k:int=4):
        """
        This method for for searching in the vector store.
        """
        docs = self.vector_db.similarity_search_with_score(query, k)
        
        return docs



    
