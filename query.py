from src.query.query_engine import get_qa_chain

if __name__ == "__main__":

    qa_chain = get_qa_chain()

    print("\n Legal RAG Assistant Ready!")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask your question: ")

        if query.lower() == "exit":
            break

        result = qa_chain.invoke(query)

        print("\n Answer:\n")
        print(result["answer"])

        print("\n Sources:\n")

        if not result["sources"]:
            print("No sources found.")
        else:
            for src in result["sources"]:
                print(f"Document: {src['source']}")
                print(f"Page: {src['page']}")
                print(f"Retrieved Text: \"{src['snippet']}\"")
                print("-" * 50)

        print("\n" + "="*50 + "\n")