class RAGSystem:
    def __init__(self, store):
        self.store = store

    def query(self, question, k=5):
        # Retrieve more candidates than needed to filter for quality
        results = self.store.search(question, k=max(k * 2, 10))
        
        # Filter out very short results (likely headers or incomplete chunks)
        filtered_results = [r for r in results if len(r.split()) >= 30]
        
        # If filtering removed too many results, use originals
        if len(filtered_results) < k:
            filtered_results = results
        
        # Take top k results
        final_results = filtered_results[:k]
        
        context = "\n\n".join(final_results)  # Better separation between chunks

        response = {
            "question": question,
            "context": context,
            "chunks": final_results
        }
        return response

