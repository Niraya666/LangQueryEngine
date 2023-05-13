from query_engine.query_engine import QueryEngine
from vector_store.vector_store import FAISSVectorStore
from models.llm_loader import llm_loader
from models.embedding_model_loader import embedding_loader 

import gradio as gr
from models.constant import SUPPORTED_LLMS, SUPPORTED_EMBEDDINGS
global query_engine
query_engine = None
def load_models_and_create_query_engine(llm_name, embeddings_name, vector_store_path, vector_store_name):
    try:
        # Load a specific language model
        LLM = llm_loader(llm_name)

        # Load a specific embeddings model
        embeddings = embedding_loader(embeddings_name)

        # Initialize a FAISSVectorStore instance with the loaded embeddings model
        vector_store = FAISSVectorStore(embeddings)
        
        # Load the vector database from a specific location
        vector_store.load_vector_db(vector_store_path, vector_store_name)

        # Initialize a QueryEngine instance
        query_Engine = QueryEngine(LLM, embeddings, vector_store)
        
        return query_Engine
    except Exception as e:
        print(f"An error occurred while loading models and creating the query engine: {e}")
        return None






def ask_question(llm_name, embeddings_name, vector_store_path, vector_store_name, query):
    global query_engine
    try:
        if query_engine is None:
            query_engine = load_models_and_create_query_engine(llm_name, embeddings_name, vector_store_path, vector_store_name)

        result = query_engine.ask_question_with_source(query)
        return result["answer"], query_engine.get_chat_history, result["source_documents"][0]
    except Exception as e:
        print(f"An error occurred while asking a question: {e}")
        return "Sorry, I couldn't process your question.", [], []

def create_gradio_interface():
    llm_dropdown = gr.inputs.Dropdown(choices=SUPPORTED_LLMS, label="Language Model")
    embeddings_dropdown = gr.inputs.Dropdown(choices=SUPPORTED_EMBEDDINGS, label="Embeddings Model")
    vector_store_path = gr.inputs.Textbox(lines=1, placeholder='./data/', label="Vector Store Path")
    vector_store_name = gr.inputs.Textbox(lines=1, placeholder='pdf', label="Vector Store Name")
    query_input = gr.inputs.Textbox(lines=2, placeholder='Enter your question here', label="Question")
    
    gr.Interface(fn=ask_question,
                 inputs=[llm_dropdown, embeddings_dropdown, vector_store_path, vector_store_name, query_input],
                 outputs=["text", "text", "text"],
                 output_names=["Answer", "Chat History", "Source Documents"],
                 output_processors=[None, lambda chat_history: "\n".join([f"{q}: {a}" for q, a in chat_history]), lambda docs: "\n\n".join([str(doc) for doc in docs])],
                 ).launch(debug=True,share=True)

if __name__ == '__main__':
    create_gradio_interface()

    # iface = gr.Interface(fn=ask_question, inputs="text", outputs="text")
    # iface.launch(debug=True,share=True)
