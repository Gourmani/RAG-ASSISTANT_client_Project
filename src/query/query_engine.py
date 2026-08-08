import os
from dotenv import load_dotenv

load_dotenv()

from src.vectorstore.qdrant_store import get_qdrant_client, create_vector_store
from src.embeddings.embedder import get_embedding_model
from src.llm.openrouter_client import get_llm

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_qa_chain():
    #  Setup
    embeddings = get_embedding_model()
    client = get_qdrant_client()
    vector_store = create_vector_store(client, embeddings)
    llm = get_llm()

    #  Prompt
    prompt = PromptTemplate.from_template(
        """You are a legal expert assistant.

Use ONLY the given context to answer.

IMPORTANT:
- Give a COMPLETE explanation
- Include key legal principles, facts, and judgment
- If answer not found, say: "The answer is not available in the provided documents."

Context:
{context}

Question:
{question}

Detailed Answer:"""
    )

    #  Retrieval
    def get_docs(query):
        docs = vector_store.similarity_search(query, k=15)
        return docs

    #  Format
    def format_docs(docs):
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
            }
            for doc in docs
        ]

    #  Generate Answer (FIXED INDENTATION)
    def generate_answer(inputs):
        docs = inputs["context"]
        question = inputs["question"]

        if not docs:
            return {
                "answer": "The answer is not available in the provided documents.",
                "sources": []
            }

        # Remove duplicates
        seen = set()
        unique_docs = []

        for doc in docs:
            key = (doc["source"], doc["page"])
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        docs = unique_docs

        # Build context
        context_text = "\n\n".join(doc["content"] for doc in docs)

        # Create prompt
        prompt_text = prompt.format(
            context=context_text,
            question=question
        )

        # LLM call
        response = llm(prompt_text)
        answer = response if isinstance(response, str) else response.content

        # Reject weak answers
        if len(answer.strip()) < 50:
            return {
                "answer": "The answer is not available in the provided documents.",
                "sources": []
            }

        # Build sources
        sources = []
        for doc in docs:
            snippet = " ".join(doc["content"].split())[:300]

            sources.append({
                "source": doc.get("source", "Unknown Document"),
                "page": doc.get("page", "Unknown Page"),
                "snippet": snippet
            })

        return {
            "answer": answer,
            "sources": sources
        }

    #  Chain (CORRECT)
    chain = (
        {
            "context": RunnableLambda(get_docs) | format_docs,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(generate_answer)
    )

    return chain