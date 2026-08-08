import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs(data_path="data"):
    documents = []

    if not os.path.exists(data_path):
        raise ValueError(f"Data folder not found: {data_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(data_path, file)

            print(f"Loading: {file}")

            loader = PyMuPDFLoader(file_path)
            docs = loader.load()

            #  Split into chunks
            docs = splitter.split_documents(docs)

            #  Add metadata
            for doc in docs:
                doc.metadata["source"] = file
                doc.metadata["page"] = doc.metadata.get("page", 0) + 1

            documents.extend(docs)

    return documents