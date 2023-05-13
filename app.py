import os
from typing import Optional, Tuple

import gradio as gr
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from threading import Lock
from langchain.llms import AI21

from query_engine.query_engine import QueryEngine
from vector_store.vector_store import FAISS_vector_store
from langchain.embeddings import CohereEmbeddings


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    # llm = OpenAI(temperature=0)
    import os
    from environs import load_dotenv
    load_dotenv('.env')
    llm = AI21(ai21_api_key=os.environ['AI21_API_KEY'])
    chain = ConversationChain(llm=llm)
    return chain
def load_query_engine():
    import os
    from environs import load_dotenv
    load_dotenv('.env')

    LLM = AI21(ai21_api_key=os.environ['AI21_API_KEY'])
    embeddings = CohereEmbeddings(cohere_api_key=os.environ['COHERE_API_KEY'])

    vector_store = FAISS_vector_store(embeddings)
    vector_store.load_local_vector_db('./data/', 'pdf')

    # result = vector_store.search('what is faceNet')
    query_Engine = QueryEngine(LLM, embeddings, vector_store)

    return query_Engine


class ChatWrapper:

    def __init__(self):
        self.query_Engine = load_query_engine()
        self.lock = Lock()
    def __call__(
        self, inp: str, history: Optional[Tuple[str, str]], 
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
           
            output = self.query_Engine.ask_question_with_source(inp)['answer']
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "Hi! How's it going?",
            "What should I do tonight?",
            "Whats 2 + 2?",
        ],
        inputs=message,
    )

    gr.HTML("Demo application of a LangChain chain.")

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    state = gr.State()
    # agent_state = gr.State()

    submit.click(chat, inputs=[message, state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state], outputs=[chatbot, state])

    

block.launch(debug=True,share=True)