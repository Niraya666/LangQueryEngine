from query_engine.query_engine import QueryEngineWithMemory
from vector_store.vector_store import FAISSVectorStore
from memory.chat_memory import chatMemory

# connection_string = 'mongodb://127.0.0.1:27017/'
from models.llm_loader import llm_loader
from models.embedding_model_loader import embedding_loader

if __name__ == '__main__':
    
    # from langchain.llms import AI21
    # from langchain.embeddings import CohereEmbeddings
    # import os
    # from environs import load_dotenv
    # load_dotenv('.env')

    # connection_string = os.environ['MONGO_CONNECTION']

    # LLM = AI21(ai21_api_key=os.environ['AI21_API_KEY'])
    # embeddings = CohereEmbeddings(cohere_api_key=os.environ['COHERE_API_KEY'])

    # vector_store = FAISS_vector_store(embeddings)
    # vector_store.load_local_vector_db('./data/', 'test')

    # Chatmemory = chatMemory(connection_string, 'test-session','ConversationSummaryMemory', llm=LLM)
    # # Chatmemory.clear_chat_memory()
    # # Chatmemory = chatMemory(connection_string, 'test-session', 'VectorStoreRetrieverMemory',embeddings = embeddings)

    # # # print(type(Chatmemory.memory))
    # # # message_history = MongoDBChatMessageHistory(
    # # # 	connection_string=connection_string, session_id="test-session")
    # # # memory = ConversationBufferMemory(
	# # # 	    memory_key="chat_history", chat_memory=message_history, return_messages=True)
    # # # print(type(memory))
    # query_Engine = QueryEngineWithMemory(LLM, embeddings, vector_store, Chatmemory.memory)

    # # # # result = vector_store.search('what is faceNet')
    # # query_Engine = QueryEngine(LLM, embeddings, vector_store)
    # result = query_Engine.ask_question_with_source('What is my name?')
    
    # print(result)

    # print(Chatmemory.load_chat_memory_json())
    embeddings = embedding_loader('Cohere')

    vector_store = FAISSVectorStore(embeddings)
    vector_store.load_vector_db('./data/', 'test')
    print(vector_store.search('what is this paper about?'))
