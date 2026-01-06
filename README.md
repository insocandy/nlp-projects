# 🦜🔗 Conversational RAG with LangChain & Llama-3

A Retrieval-Augmented Generation (RAG) pipeline that allows users to chat with web documents. Built with **LangChain**, **Groq (Llama-3)**, and **ChromaDB**.

## 🚀 Features
* **Web Scraping:** Loads and parses technical blog posts.
* **Vector Search:** Uses HuggingFace embeddings + ChromaDB for semantic retrieval.
* **Conversational Memory:** Remembers context for follow-up questions.
* **Llama-3 Power:** Uses Groq for ultra-fast inference.

## 🛠️ Installation

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/conversational-rag.git](https://github.com/YOUR_USERNAME/conversational-rag.git)
    cd conversational-rag
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Keys:**
    Create a `.env` file and add your keys:
    ```text
    GROQ_API_KEY=your_key_here
    HF_TOKEN=your_token_here
    ```

4.  **Run the Notebook:**
    ```bash
    jupyter notebook first_proj.ipynb
    ```
