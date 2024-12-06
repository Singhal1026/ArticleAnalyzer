# Article Analyzer 📰

This **Article Analyzer** is a powerful application that allows users to input URLs of articles, process and embed the content into a vector store, and retrieve relevant information by asking natural language questions. The tool leverages **HuggingFace embeddings, FAISS vector storage, and the Google Gemini LLM** for providing precise answers with source attribution.

## Features

- **URL Input & Processing**: Input multiple URLs to extract and preprocess the text content.
- **Document Embedding**: Transform the text into embeddings using **HuggingFace's sentence-transformers**.
- **Efficient Storage**: Store embeddings in a highly optimized FAISS vector database.
- **Natural Language Q&A**: Ask questions in plain English and get context-aware answers.
- **Source Attribution**: View sources for the answers to ensure credibility.


## TechStack

### Languages & Frameworks
- Python
- Streamlit

### Libraries
- LangChain
- HuggingFace
- FAISS
- Unstructured

## Installation

### Prerequisites
1. Python 3.8 or higher.
2. A valid Google Generative AI API key.

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Singhal1026/ArticleAnalyzer.git
   cd ArticleAnalyzer
   ```

2. **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Application**:
    ```bash
    streamlit run app.py
    ```

## Access the Application

You can access the deployed application at the following [URL](https://huggingface.co/spaces/yash1026/Article-Analyzer):