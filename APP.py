#python -m streamlit run APP.py     

import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
with st.sidebar:
    st.title("LLM PDF APP")
    st.markdown('''
    ## About
    This app is a demo of the LLM PDF app.
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://langchain.com/)
    - [OpenAI](https://openai.com/) LLM model
    ''')

    add_vertical_space(5)
    st. write('''Made by Jason Hao.
              haomengqi12138@gmail.com''')


def main():
    st.header ("LLM PDF APP")
    pdf = st.file_uploader("Upload a PDF file", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunk = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]


        
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
            st.write("embedding loaded from disk")
        else:
            
            embeddings = OpenAIEmbeddings(os.getenv("OPENAI_API_KEY"))
            vector_store = Chroma.from_documents(chunk, embedding = embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
            st.write("embeddings computation completed")

        query = st.text_input("Enter your question here")
        st.write("You entered: ", query)
        if query:
            docs = vector_store.similarity_search(query,k=5 )
            llm = OpenAI(temperature= 0, openai_api_key=openai_api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents = docs, question = query)
            st.write(response)

if __name__ == "__main__":
    main()
