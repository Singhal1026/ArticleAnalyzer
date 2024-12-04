import os
import streamlit as st
import pickle
import time
import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title("Article Analyzer 📰") 

st.sidebar.title("Article URLs")

# Collect URLs from sidebar inputs
urls = [url for i in range(3) if (url := st.sidebar.text_input(f"URL {i+1}"))]
process_urls = st.sidebar.button("Process URLs") 
file_path = "vector_store.pkl"

main_placeholder = st.empty()


if process_urls and urls:
    try:
        # Validate if there are valid URLs
        if not any(urls):
            main_placeholder.error("Please enter at least one valid URL.")
        else:
            # Load data
            # st.write("Processing URLs...")
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.write("Loading data from URLs...🔃🔃🔃")
            data = loader.load()

            # Split data
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","],
                chunk_size=1000
            )
            main_placeholder.write("Splitting data...🔃🔃🔃")
            docs = splitter.split_documents(data)

            # Create embeddings and store them in FAISS index
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            main_placeholder.write("Embedding data...🔃🔃🔃")
            vector_store = FAISS.from_documents(docs, embeddings)

            # Save the FAISS index to a pickle file
            if os.path.exists(file_path):
                st.warning("Overwriting existing vector store file.")
            with open(file_path, 'wb') as f:
                pickle.dump(vector_store, f)

            main_placeholder.success("Processing complete! Data embedded and saved.")
    except Exception as e:
        main_placeholder.error(f"An error occurred: {e}")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model = 'gemini-pro', google_api_key = GEMINI_API_KEY)


query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vector_store = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever=vector_store.as_retriever())
            result = chain({'question': query}, return_only_outputs=True)
            # {"answer": "The answer to the question", "sources": [{"url": "https://source1.com", "score": 0.9}, {"url": "https://source2.com", "score": 0.8}]}
            st.header("Answer")
            st.subheader(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)


