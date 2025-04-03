# rag-system
Implement a Retrieval-Augmented Generation (RAG) system with LangChain to make an LLM retrieve and use external knowledge dynamically.

## Set up
```bash
python -m venv venv
source venv/bin/activate # MacOS/Linux
venv\Scripts\activate # Windows
pip install langchain langchain-community langchain-huggingface langchain_huggingface transformers faiss-cpu sentence-transformers streamlit torch
```
## Run

```bash
streamlit run app.py --server.fileWatcherType=none
```