import streamlit as st
#import langFlow as lf
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
import os


#load_dotenv() # load the environment variables
def get_response(user_input):
    retriever_chain=get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain=get_conversation_rag_chain(retriever_chain)
    response=conversation_rag_chain.invoke({
        "chat_history":st.session_state.chat_history,
        "input":user_query
    })
    return response ['answer']
def get_vectorestore_from_url(url):
    # get the documents from the website and split them into chuncks
    loader=WebBaseLoader(url)  
    document=loader.load() 
    text_splitter=RecursiveCharacterTextSplitter()
    document_chuncks=text_splitter.split_documents(document)
    # get the vector store
    vector_store=Chroma.from_documents(document_chuncks, OpenAIEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store): #ok
    
    llm=ChatOpenAI() 
    llm.openai_api_key="sk-proj-3GIXq53h9AB1FnRIrEpmT3BlbkFJD6m4veEyJIOgUxs39Gtg"   
    
    # llm.openai_api_key = os.getenv("OPENAI_API_KEY")
     

    retriever=vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder (variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "given the above conversation, generate a search query to look up in order to get information revant to the conversation "),

    ])
    retriever_chain=create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
def get_conversation_rag_chain(retriever_chain): #ok
    
    llm=ChatOpenAI()
    llm.openai_api_key="sk-proj-3GIXq53h9AB1FnRIrEpmT3BlbkFJD6m4veEyJIOgUxs39Gtg"
    
    prompt=ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain=create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# app config
st.set_page_config(page_title="Chatbot", page_icon=":robot:")
st.title("Chatbot")
#if "chat_history" not in st.session_state:
    #st.session_state.chat_history = [
        #AIMessage(content="Ola! Eu sou a secretária da CMF, em que posso lhe ajudar?"),
    #]



#sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
if website_url is  None or website_url == "":
    st.info("Entre com o Website")
else:
    #session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Ola! Eu sou a secretária da CMF, em que posso lhe ajudar?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store=get_vectorestore_from_url(website_url)    
    
    # create conversation chain
    retriever_chain=get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain=get_conversation_rag_chain(retriever_chain)
    #user input
    user_query = st.chat_input(placeholder="Digite sua mensagem...")
    if user_query is not None and user_query != "":
        response=get_response(user_query)        
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        #retrieved_documents=retriever_chain.invoke({
            #"chat_history":st.session_state.chat_history,
            #"input":user_query
        #})
        #st.write(retrieved_documents)
        #with st.chat_message("human"):
            #st.write(user_query)
        #with st.chat_message("AI"):
            #st.write(response)
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.write(message.content)
        

    