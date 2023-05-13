from langchain.chains import ConversationalRetrievalChain

class QueryEngine:
    """
    The `QueryEngine` class provides methods for semantic search and question-answering, using a specified language model, embeddings model, and vector store. The class also maintains a chat history.

    Attributes:
    - llm: The language model.
    - embeddings: The embeddings model.
    - vectorstore: The vector store.
    - chain: The ConversationalRetrievalChain object.
    - chat_history: The chat history.

    Methods:
    - semantic_search: Perform semantic search with a given query.
    - load_qa_chain: Load the question-answering chain from the language model and vector store.
    - get_chat_history: Return the chat history.
    - ask_question_with_source: Ask a question and append the query and answer to the chat history.
    """
    def __init__(self, LLM, 
    				Embeddings,
    				Vectorstore):
        self.llm = LLM
        self.embeddings = Embeddings 
        self.vectorstore = Vectorstore

        self.chain = self.load_qa_chain()
        self.chat_history = []

    def semantic_search(self, query:str, k:int=4):
        return self.vectorstore.search(query, k)
    def load_qa_chain(self):

        chain = ConversationalRetrievalChain.from_llm(self.llm, self.vectorstore.vector_db.as_retriever(), return_source_documents=True)

        return chain
    @property 
    def get_chat_history(self):
        return self.chat_history


    def ask_question_with_source(self, query):
        # vectordbkwargs = {"search_distance": 0.9}
        
        result = self.chain({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        return result



class QueryEngineWithMemory:
    """
    The `QueryEngineWithMemory` class is an extension of `QueryEngine`, with added memory capabilities.

    Attributes:
    - memory: The chat memory.

    Methods:
    - load_qa_chain: Load the question-answering chain from the language model and vector store, with memory capabilities.
    - ask_question_with_source: Ask a question without modifying the chat history.
    - get_chat_history: Return the memory object.
    """
    def __init__(self,
                    LLM, 
                    Embeddings,
                    Vectorstore,
                    Memory):
        self.llm = LLM
        self.embeddings = Embeddings 
        self.vectorstore = Vectorstore
        
        self.memory = Memory
        self.chain = self.load_qa_chain()

    def load_qa_chain(self):

        chain = ConversationalRetrievalChain.from_llm(self.llm, self.vectorstore.vector_db.as_retriever(), memory = self.memory)

        return chain

    def ask_question_with_source(self, query):
        # vectordbkwargs = {"search_distance": 0.9}
        
        result = self.chain({"question": query})
        # self.chat_history.append((query, result["answer"]))
        return result

    def get_memory(self):
        return self.memory



