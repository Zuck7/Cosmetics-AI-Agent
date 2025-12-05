from rag.pdf_loader import load_pdfs
from rag.vector_store import VectorStore
from rag.rag_system import RAGSystem
from summarization_agent.sum_agent import SummarizationAgent

# 1. Load documents
docs = load_pdfs("data/cosmetics")

# 2. Build vector store
store = VectorStore()
store.build(docs)

# 3. RAG System
rag = RAGSystem(store)

# 4. Summarization Agent
summarizer = SummarizationAgent()

# User query
query = "What ingredients are commonly used for anti aging skin care?"

# 5. Retrieve context
rag_output = rag.query(query, k=4)

# 6. Summarize
summary = summarizer.summarize(rag_output["context"])

print("QUERY:", query)
print("\nRELEVANT CONTEXT:\n", rag_output["context"])
print("\nSUMMARY:\n", summary)
