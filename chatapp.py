from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

import os
import openai

import streamlit as st
from streamlit_chat import message


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']


# load vectordb 
embedding = OpenAIEmbeddings(model = "text-embedding-ada-002")
vectordb = Chroma(persist_directory="docs/chroma",
                  embedding_function=embedding)

#retriever
retriever = vectordb.as_retriever(search_type = "similarity",
                                  search_kwargs = {'k':2})


# create buffer memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


# create chat llm model
model = "gpt-3.5-turbo"
llm = ChatOpenAI(model = model,
                temperature = 0)

chat = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = retriever,
    chain_type = "stuff",
    memory = memory
)


# Page title
st.set_page_config(page_title='🦜🔗 Northwind Chat APP')
st.title('🦜🔗Northwind Chat Assistant')


# Initialize chat object
if "chat_obj" not in st.session_state:
    st.session_state.chat_obj = chat

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {'role':'assistant',
         'content':"Hello! I am an AI assitant. I am here to help you with questions on Northwind Health Plus Benefits.\
                    How can I help you today?"}
    ]

chat_container = st.container()
input_container = st.container()


#response = "thank you"

def on_input_change():
    user_query = st.session_state.user_input
    st.session_state.messages.append({"role": "user", "content":user_query})

    # get response from llm
    chat = st.session_state.chat_obj
    response = chat.run(user_query)

    st.session_state.messages.append({'role':'assistant', 'content': response})
    # update chat
    st.session_state.chat_obj = chat


with chat_container:
            
    # Display chat messages from history on app rerun
    if st.session_state['messages']:
        for i, msg in enumerate(st.session_state.messages):
            if msg['role'] == 'user':
                 message(msg['content'], is_user = True, key = str(i) + '_user')
            else:
                message(msg['content'], key = str(i))

with input_container:
    user_query = st.text_input("", on_change = on_input_change,  key = 'user_input')
    

    




