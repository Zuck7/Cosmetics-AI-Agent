from rag.pdf_loader import load_pdfs
from rag.cosing_loader import load_cosing
from rag.vector_store import VectorStore
from rag.rag_system import RAGSystem
from summarization_agent.sum_agent import SummarizationAgent
from planner_agent.planner import PlannerAgent

# ============================================================================
# SETUP: Initialize all agents and components
# ============================================================================

print("Initializing Cosmetics AI Agent System...")
print("=" * 60)

# 1. Load documents from knowledge base: PDFs + the open EU CosIng ingredient database
print("\n[1/4] Loading PDF documents and CosIng ingredient database...")
pdf_docs = load_pdfs("data/cosmetics")
cosing_docs = load_cosing("data/cosing/COSING_Ingredients-Fragrance.Inventory_v2.csv")
docs = pdf_docs + cosing_docs
print(f"✓ Loaded {len(pdf_docs)} PDF document(s) and {len(cosing_docs)} CosIng ingredient record(s)")

# 2. Build vector store for RAG
print("\n[2/4] Building vector store...")
store = VectorStore()
store.build(docs)
print(f"✓ Vector store built with {len(store.text_chunks)} chunks")

# 3. Initialize RAG System
print("\n[3/4] Initializing RAG system...")
rag = RAGSystem(store)
print("✓ RAG system ready")

# 4. Initialize Summarization Agent
print("\n[4/4] Initializing Summarization Agent...")
summarizer = SummarizationAgent()
print("✓ Summarization agent ready")

# 5. Initialize Planner Agent (Top-level Coordinator)
print("\n[MAIN] Initializing Planner Agent (Coordinator)...")
planner = PlannerAgent(
    rag_system=rag,
    summarizer=summarizer
)
print("✓ Planner agent ready")

print("\n" + "=" * 60)
print("System initialization complete!")
print("=" * 60)

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Example 1: Basic query with planner orchestration
print("\n\n" + "=" * 60)
print("EXAMPLE 1: Basic Query")
print("=" * 60)

query1 = "What ingredients are commonly used for anti aging skin care?"

# The planner automatically handles:
# - Query decomposition
# - Retrieval from RAG
# - Summarization
result1 = planner.coordinate(query1)

# Display formatted output
print(planner.format_output(result1))


# Example 2: Another query
print("\n\n" + "=" * 60)
print("EXAMPLE 2: Safety Query")
print("=" * 60)

query2 = "What are the safety concerns with retinol in cosmetics?"

result2 = planner.coordinate(query2)
print(planner.format_output(result2))


# Example 3: Complex query
print("\n\n" + "=" * 60)
print("EXAMPLE 3: Complex Query")
print("=" * 60)

query3 = "Compare the effectiveness of vitamin C and niacinamide for skin brightening"

result3 = planner.coordinate(query3)
print(planner.format_output(result3))


# ============================================================================
# DIRECT ACCESS TO RESULTS
# ============================================================================

print("\n\n" + "=" * 60)
print("ACCESSING STRUCTURED RESULTS")
print("=" * 60)

# You can access individual components of the result
print("\nAnswer:", result1['answer'])
print("\nRetrieved Chunks:", result1['metadata']['retrieved_chunks'])

# Access supporting evidence
print("\n\nSupporting Evidence (first 200 chars of each):")
for i, chunk in enumerate(result1['supporting_evidence'][:3], 1):
    print(f"\n[Chunk {i}]")
    print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

print("\n\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
