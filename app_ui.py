import streamlit as st
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Local Offline RAG Assistant", page_icon="📄")
st.title("📄 Offline Document Assistant (Local RAG)")
st.caption("Powered 100% locally by Ollama, ChromaDB, and LangChain")

st.sidebar.header("1. Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])

if uploaded_file:
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    if st.sidebar.button("Index Document"):
        with st.spinner("Processing and vectorizing document locally..."):
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            chunks = text_splitter.split_documents(documents)
            
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            
            st.session_state["retriever"] = vectorstore.as_retriever(search_kwargs={"k": 2})
            st.sidebar.success("Document successfully indexed!")

st.header("2. Ask Your Document")
user_query = st.text_input("Enter your question below:")

if user_query:
    if "retriever" not in st.session_state:
        st.warning("Please upload and index a document in the sidebar first!")
    else:
        with st.spinner("Generating answer locally..."):
            llm = ChatOllama(model="qwen2.5:0.5b", temperature=0)
            
            system_prompt = (
                "Answer the question strictly using the provided context below. "
                "If the answer is not in the context, say 'Information not available in document.'\n\n"
                "Context:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(st.session_state["retriever"], question_answer_chain)
            
            response = rag_chain.invoke({"input": user_query})
            
            st.subheader("Answer:")
            st.write(response["answer"])