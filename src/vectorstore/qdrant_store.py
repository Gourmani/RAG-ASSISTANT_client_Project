import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()


def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    client = QdrantClient(
        url=url,
        api_key=api_key,
         timeout=120
    )
    return client


def create_vector_store(client, embeddings, collection_name="legal_rag"):
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    return vector_store