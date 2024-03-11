import os
import json
import streamlit as st
from tempfile import NamedTemporaryFile

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
from langchain_community.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

key1 = os.environ.get("OPENAI_API_KEY")

# load and split the documents using pypdf and text splitters
def load_and_split_document(uploaded_file):
    """Loads and splits the document into pages."""

    bytes_data = uploaded_file.read()
    with NamedTemporaryFile(delete=False) as tmp: 
        tmp.write(bytes_data)                      
        data = PyPDFLoader(tmp.name).load_and_split()
    os.remove(tmp.name)
    return data  
    
    

#create embeddings and store them in a database
def embeddings(key):
    return OpenAIEmbeddings(openai_api_key = key)

def create_database(data,embeddings):

    db = FAISS.from_documents(data,embeddings)
    db.save_local("faiss_database")

# create the prompt,model and chain using langchain 

prompt = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

def get_model(key,model_name = "gpt-3.5-turbo"):
    chat_model = ChatOpenAI(openai_api_key = key,model_name=model_name)
    return chat_model

def response(database,model,question):
    
    prompt_val = ChatPromptTemplate.from_template(prompt)
    retreiver = database.as_retriever()
    parser = StrOutputParser()
    chain = (
        {'context': retreiver,'question':RunnablePassthrough()}
        | prompt_val
        | model
        | parser
    )
    ans = chain.invoke(question)
    return ans

def main():
    st.set_page_config("Chat Q&A")
    st.header("Chat with your pdf documents")

    uploaded_file = st.file_uploader("Upload your document", type=["pdf"])
    
    if uploaded_file:

        
        st.header("update vector store")

        if st.button("update vectors"):
            with st.spinner("processing..."):
                data = load_and_split_document(uploaded_file)
                embed = embeddings(key1)
                create_database(data,embed)
                st.success("done")
                    
        question = st.text_input("Ask question about your documents")

        if question:

            if st.button("OpenAI result"):
                with st.spinner("processing..."):
                    embed = embeddings(key1)
                    database = FAISS.load_local("faiss_database",embed,allow_dangerous_deserialization=True)
                    model = get_model(key1)
                    st.write(response(database,model,question))
                    st.success("done")

if __name__=="__main__":
    main()

    






