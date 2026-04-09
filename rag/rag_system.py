class RAGSystem:
    def __init__(self, store):
        self.store = store

    def query(self, question, k=5):
        # Retrieve source-aware chunks and diversify by PDF source where possible.
        results = self.store.search(
            question,
            k=max(k * 2, 10),
            return_metadata=True,
            diversify_sources=True,
            per_source_cap=3,
        )

        # Filter out very short results (likely headers or incomplete chunks).
        filtered_results = [r for r in results if len(r.get("text", "").split()) >= 20]

        # If filtering removed too many results, use originals.
        if len(filtered_results) < k:
            filtered_results = results

        # Take top k results.
        final_results = filtered_results[:k]
        final_chunks = [r.get("text", "") for r in final_results]
        final_sources = [r.get("source", "unknown") for r in final_results]

        context = "\n\n".join(final_chunks)  # Better separation between chunks

        response = {
            "question": question,
            "context": context,
            "chunks": final_chunks,
            "sources": final_sources,
        }
        return response

