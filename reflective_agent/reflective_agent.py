# agents/reflective_agent.py

from typing import Dict, Any, List
from validation_agent.validation_agent import ValidationReport


def reflect_on_run(report: ValidationReport) -> Dict[str, Any]:
    """
    Reflective agent: looks at the validation report and suggests improvements
    for the whole pipeline (planner, RAG, summarizer, validator).
    """
    feedback = {}
    actions: List[str] = []

    # --- High-level judgement ---
    if report.overall_passed and report.relevance_score >= 0.6:
        overall = (
            "The system performed well. The summary is reasonably aligned with the "
            "retrieved evidence and most rules were satisfied."
        )
    elif report.overall_passed:
        overall = (
            "The system barely passed. The summary meets minimum criteria, but "
            "relevance or rule coverage is weak. Future runs should focus on stronger retrieval."
        )
    else:
        overall = (
            "The system did not pass validation. Either the summary is not strongly grounded "
            "in the retrieved chunks, or several rules were violated."
        )

    # --- Planner feedback (Zuck) ---
    if report.relevance_score < 0.4:
        feedback["planner_agent"] = (
            "Review the planning step. The query decomposition or task breakdown may not be "
            "guiding the RAG system to the most relevant sections of the PDFs."
        )
        actions.append("Refine planner prompts / task decomposition for this query.")
    else:
        feedback["planner_agent"] = "Planning appears adequate for this query."

    # --- RAG system feedback ---
    if report.retrieved_chunks and report.relevance_score < 0.5:
        feedback["rag_system"] = (
            "RAG retrieval did not strongly align with the final summary. Consider improving "
            "chunking, embeddings, or similarity thresholds."
        )
        actions.append("Tune RAG retrieval parameters or try a better embedding model.")
    elif not report.retrieved_chunks:
        feedback["rag_system"] = (
            "No retrieved chunks were available. RAG system may have failed or not been called."
        )
        actions.append("Debug RAG pipeline – ensure documents are indexed and retrieved.")
    else:
        feedback["rag_system"] = "RAG retrieval seems reasonably aligned with the summary."

    # --- Summarization agent feedback ---
    if report.word_count == 0:
        feedback["summarization_agent"] = (
            "Summarization agent produced an empty summary. Investigate prompt or model failure."
        )
        actions.append("Fix summarizer to always produce a non-empty answer.")
    elif report.word_count > 250:
        feedback["summarization_agent"] = (
            "Summary is too long. Summarizer should be prompted or constrained to be shorter."
        )
        actions.append("Tighten summarization instructions to respect word limits.")
    else:
        feedback["summarization_agent"] = "Summary length is acceptable."

    # --- Validation agent feedback (YOU) ---
    failed_auto_rules = [
        r for r in report.rule_checks
        if (not r.passed) and "manual review" not in r.comment.lower()
    ]
    if not failed_auto_rules:
        feedback["validation_agent"] = (
            "Validation rules ran correctly. Consider extending automatic checks "
            "to cover more of the rules.txt content."
        )
    else:
        feedback["validation_agent"] = (
            "Some rules failed. Validation agent successfully detected issues. "
            "Update rules.txt or summarization prompts based on these failures."
        )

    # --- Suggested next steps for system evaluation ---
    if not actions:
        actions.append("Run additional test queries to stress-test the system.")
    actions.append("Document this reflection in the report as part of system evaluation.")

    return {
        "overall_assessment": overall,
        "agent_feedback": feedback,
        "suggested_actions": actions,
    }


def pretty_print_reflection(reflection: Dict[str, Any]) -> None:
    print("=" * 80)
    print("REFLECTIVE AGENT OUTPUT")
    print("=" * 80)
    print("Overall assessment:")
    print(reflection["overall_assessment"])
    print("\n--- Agent-specific feedback ---")
    for agent, fb in reflection["agent_feedback"].items():
        print(f"\n[{agent}]")
        print(fb)
    print("\n--- Suggested actions ---")
    for i, act in enumerate(reflection["suggested_actions"], start=1):
        print(f"{i}. {act}")
    print("=" * 80)
