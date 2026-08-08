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

        print("\n Source:\n")

        if not result["sources"]:
            print("The answer is not available in the provided documents.")
        else:
            # Show only top 2 sources (clean)
            for src in result["sources"][:2]:
                print(f"{src['source']}")
                print(f"Page {src['page']}")
                print("Retrieved Text:")
                print(f"\"{src['snippet']}\"")
                print()

        print("\n" + "="*50 + "\n")