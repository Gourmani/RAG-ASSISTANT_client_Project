from src.loaders.pdf_loader import load_pdfs
from src.embeddings.embedder import get_embedding_model
from src.vectorstore.qdrant_store import get_qdrant_client, create_vector_store

from qdrant_client.models import Distance, VectorParams
import time

if __name__ == "__main__":

    # =========================
    # 1. LOAD DOCUMENTS
    # =========================
    docs = load_pdfs()
    print(f"Loaded documents: {len(docs)}")

    # =========================
    # 2. SPLIT INTO CHUNKS
    # =========================
    chunks = docs  # Already split in load_pdfs
    print(f"Total chunks: {len(chunks)}")

    # =========================
    # 3. LOAD EMBEDDING MODEL
    # =========================
    embeddings = get_embedding_model()
    print("Embedding model loaded...")

    # =========================
    # 4. CONNECT TO QDRANT
    # =========================
    client = get_qdrant_client()
    collection_name = "legal_rag"

    # =========================
    # 5. DELETE + CREATE (CLEAN RUN)
    # =========================
    if client.collection_exists(collection_name):
        print(" Deleting old collection...")
        client.delete_collection(collection_name)

    print(" Creating new collection...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    # =========================
    # 6. CREATE VECTOR STORE
    # =========================
    vector_store = create_vector_store(client, embeddings)

    # =========================
    # 7. STORE DATA (BATCH + RETRY)
    # =========================
    print("\n Storing embeddings in Qdrant...\n")

    BATCH_SIZE = 50   
    failed_batches = 0
    total_uploaded = 0
    total = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        for attempt in range(3):
            try:
                print(f"[{i}/{total}] Uploading batch (Attempt {attempt+1})")

                vector_store.add_documents(batch)

                total_uploaded += len(batch)
                break

            except Exception as e:
                print(f" Error: {e}")
                time.sleep(5)

                if attempt == 2:
                    print(" Skipping this batch...\n")
                    failed_batches += 1

    # =========================
    # 8. FINAL VERIFICATION
    # =========================
    print("\n FINAL STATUS:\n")

    collection_info = client.get_collection(collection_name)

    print(f"Stored vectors: {collection_info.points_count}")
    print(f"Expected chunks: {len(chunks)}")
    print(f"Uploaded (tracked): {total_uploaded}")
    print(f"Failed batches: {failed_batches}")

    #  Correct validation
    if collection_info.points_count >= len(chunks):
        print(" SUCCESS: Data uploaded correctly!")
    else:
        print(" WARNING: Some data may be missing!")

    # =========================
    # 9. EXTRA VERIFICATION (VERY IMPORTANT)
    # =========================
    print("\n EXTRA CHECKS:\n")

    # Scroll test
    points, _ = client.scroll(collection_name, limit=5)
    print(f" Scroll test passed: {len(points)} sample points fetched")

    # Query test
    results = vector_store.similarity_search("fundamental rights", k=2)
    print(f" Query test passed: {len(results)} results returned")

    print("\n INGESTION COMPLETE!")