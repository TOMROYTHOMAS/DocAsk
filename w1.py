import streamlit as st
import os
import shutil
import chromadb
import chromadb.config  # Ensure Chroma settings are used correctly
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

# Streamlit UI
st.title("RAG-based Q&A Chatbot with Ollama")

# File Upload
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    file_path = os.path.join("uploaded." + file_extension)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved as {file_path}")
    
    # Load Document
    docs = []
    if file_extension == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif file_extension == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            docs = [Document(page_content=text)]
    
    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    
    # Safely remove existing ChromaDB index
    if os.path.exists("db"):
        try:
            shutil.rmtree("db")
        except PermissionError:
            st.error("Error: Could not delete existing ChromaDB index. Please restart the application and try again.")
    
    # Use DuckDB+Parquet to avoid SQLite issues
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vector_store = Chroma.from_documents(
        split_docs, 
        embedding_model, 
        persist_directory="db",
        client_settings=chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",  # Use DuckDB instead of SQLite
            persist_directory="db"
        )
    )
    vector_store.persist()
    st.success("Document successfully indexed into ChromaDB!")
    
    # Initialize LLM
    llm = Ollama(model="llama3.2")
    
    # Create RAG-based retrieval function
    def create_rag():
        return RetrievalQA.from_llm(llm, retriever=vector_store.as_retriever())
    
    # Initialize QA chain
    qa_chain = create_rag()
    
    # User Question Input
    user_question = st.text_input("Ask a question about the document:")
    if user_question:
        response = qa_chain.run(user_question)
        st.write("### Answer:")
        st.write(response)
