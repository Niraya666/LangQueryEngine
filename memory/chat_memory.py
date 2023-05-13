from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
from langchain.schema import _message_to_dict
from langchain.memory import ConversationBufferWindowMemory
import json
from langchain.memory import ConversationSummaryMemory
from langchain.memory import VectorStoreRetrieverMemory



class chatMemory:
    """
    The `chatMemory` class provides methods for managing the memory of a chat session, which includes initializing, loading, and clearing the memory. This class supports different types of memory models and allows the chat history to be stored in a MongoDB database.

    Attributes:
    - connection_string (str): The MongoDB connection string.
    - session_id (str): The session ID.
    - memory_type (str): The type of memory model to use. The default is 'ConversationBufferMemory'.
    - message_history (MongoDBChatMessageHistory): The MongoDB chat message history object.
    - memory (ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, VectorStoreRetrieverMemory): The memory model object.

    Methods:
    - _init_memory: Initializes the memory model based on the specified memory_type.
    - _init_vector: Initializes the vector store for the 'VectorStoreRetrieverMemory' type.
    - chat_memory: Returns the memory model object.
    - load_chat_memory: Returns the chat messages from the memory.
    - load_chat_memory_json: Returns the chat messages from the memory in JSON format.
    - clear_chat_memory: Clears the chat memory.
    """
    def __init__(self, 
        connection_string, 
        session_id, 
        memory_type = 'ConversationBufferMemory',
        **kwargs
        ):
        self.connection_string = connection_string
        self.session_id = session_id
        self.message_history = MongoDBChatMessageHistory(
                connection_string=connection_string, session_id=session_id
            )
        self.memory_type = memory_type
        self._init_memory(**kwargs)
        
    def _init_memory(self,**kwargs):
        if self.memory_type == 'ConversationBufferMemory':
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                chat_memory=self.message_history, return_messages=True,
                **kwargs
                )
        elif self.memory_type=='ConversationBufferWindowMemory':
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history", 
                chat_memory=self.message_history, return_messages=True,
                **kwargs
                )

        elif self.memory_type=='ConversationSummaryMemory':
            self.memory = ConversationSummaryMemory(
                memory_key="chat_history", 
                chat_memory=self.message_history, return_messages=True,
                **kwargs
                )
        elif self.memory_type=='VectorStoreRetrieverMemory':
            self._init_vectore(**kwargs)
            retriever = self.memory_vectorstore.as_retriever(search_kwargs=dict(k=1))
            self.memory = VectorStoreRetrieverMemory(retriever=retriever)
        else:
            raise ValueError(f"Unknown memory_type: {self.memory_type}")

    def _init_vectore(self,embeddings):
        import faiss
        from langchain.docstore import InMemoryDocstore
        from langchain.vectorstores import FAISS

        embedding_size = 4096 # Dimensions of the Embeddings
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = embeddings.embed_query
        self.memory_vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

    @property 
    def chat_memory(self):
        return self.memory
    
    def load_chat_memory(self):
        return self.memory.chat_memory.messages

    def load_chat_memory_json(self):
        messages = self.memory.chat_memory.messages
        messages_json = json.dumps([_message_to_dict(msg) for msg in messages])
        return messages_json

    def clear_chat_memory(self):
        self.memory.chat_memory.clear()