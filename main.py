# main.py

from validation_agent.validation_agent import validate_summary, pretty_print_report
from reflective_agent.reflective_agent import reflect_on_run, pretty_print_reflection


from rag.pdf_loader import load_pdfs
from rag.vector_store import VectorStore
from rag.rag_system import RAGSystem
from summarization_agent.sum_agent import SummarizationAgent


def run_full_pipeline(user_query: str):
   

    #Load PDFs
    documents = load_pdfs("data/cosmetics")  # <-- make sure his PDFs are inside data/

    #Build Vector Store
    store = VectorStore()
    store.build(documents)

    #Run RAG Retrieval
    rag = RAGSystem(store)
    rag_result = rag.query(user_query, k=5)

    retrieved_chunks = rag_result["chunks"]
    context_text = rag_result["context"]

    #Summarize
    summarizer = SummarizationAgent()
    summary = summarizer.summarize(context_text, max_len=120)

    return retrieved_chunks, summary


def main():
    user_query = "Summarize the main safety considerations for cosmetic products."
    print(f"USER QUERY: {user_query}\n")

    #RAG + Summarization 
    chunks, summary = run_full_pipeline(user_query)

    #Validation Agent
    report = validate_summary(
        query=user_query,
        summary=summary,
        retrieved_chunks=chunks,
        min_relevance=0.4,
        min_word_count=20
    )

    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)
    print(pretty_print_report(report))

    #Reflective Agent
    reflection = reflect_on_run(report)

    print("\n" + "="*70)
    print("REFLECTIVE AGENT OUTPUT")
    print("="*70)
    print(pretty_print_reflection(reflection))


if __name__ == "__main__":
    main()
