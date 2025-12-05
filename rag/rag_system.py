class RAGSystem:
    def __init__(self, store):
        self.store = store

    def query(self, question, k=5):
        results = self.store.search(question, k)
        context = "\n".join(results)

        response = {
            "question": question,
            "context": context,
            "chunks": results
        }
        return response
