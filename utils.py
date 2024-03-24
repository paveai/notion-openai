import streamlit as st
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from ingest import load_faiss_index # Import load_faiss_index from ingest.py
        
@st.cache_resource
def load_chain(openai_key):
    print("inside load_chain line 15")
    """
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """
    
    embeddings = OpenAIEmbeddings(api_key=openai_key)
    
    # Load OpenAI chat model
    llm = ChatOpenAI(api_key=openai_key, temperature=0)
    
    # Load our local FAISS index as a retriever
    if os.path.exists("faiss_index/index.faiss") and os.path.isdir("faiss_index"):
        print("available")
    else:
        # "faiss_index" directory does not exist, call load_faiss_index function
        load_faiss_index(openai_key)
    
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # Create memory 'chat_history' 
    memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(llm, 
                                                  retriever=retriever, 
                                                  memory=memory, 
                                                  get_chat_history=lambda h : h,
                                                  verbose=True)

    # Create system prompt
    template = """
    You are an AI assistant for answering questions about the Blendle Employee Handbook.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say 'Sorry, I don't know... 😔'.
    Don't try to make up an answer.
    If the question is not about the Blendle Employee Handbook, politely inform them that you are tuned to only answer questions about the Blendle Employee Handbook.

    {context}
    Question: {question}
    Helpful Answer:"""

    # Add system prompt to chain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

    return chain