#  Legal RAG Assistant

##  Overview

This project is a Retrieval-Augmented Generation (RAG) system that answers legal questions strictly using the provided PDF documents.

It retrieves relevant content from the documents and generates accurate answers using an LLM, along with proper citations.

The system is designed to avoid hallucination and only responds based on available document context.

---

##  Architecture

1. PDFs are loaded using PyMuPDFLoader  
2. Text is split into chunks (chunk_size=500, overlap=100)  
3. Embeddings are generated using Sentence Transformers  
4. Embeddings are stored in Qdrant vector database  
5. User query → similarity search → top-k relevant chunks retrieved  
6. OpenRouter LLM generates answer using retrieved context  
7. System returns:
   - Answer  
   - Document name  
   - Page number  
   - Retrieved text snippet  

---

##  Libraries Used

- LangChain  
- Qdrant Client  
- PyMuPDF  
- Sentence Transformers  
- OpenRouter API  
- Python-dotenv  

---

##  Embedding Model

Sentence Transformers (384-dimension embeddings via LangChain)

---

##  Assumptions

- PDFs contain readable legal text  
- Queries are related to document content  
- Top-k retrieval provides sufficient context  

---

##  How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt