from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ValidationReport:
    query: str
    summary: str
    retrieved_chunks: List[str]
    relevance_score: float
    word_count: int
    overall_passed: bool
    rule_checks: List[Dict[str, Any]]


def validate_summary(
    query: str,
    summary: str,
    retrieved_chunks: List[str],
    min_relevance: float = 0.4,
    min_word_count: int = 20,
    rules_path: str = "rules.txt",
) -> ValidationReport:
    """
    Validate the summary against basic quality checks + simple rule file.
    """
    
    words = summary.split()
    word_count = len(words)

   
    matches = 0
    for chunk in retrieved_chunks:
        for w in words:
            if w.lower() in chunk.lower():
                matches += 1
                break

    relevance_score = matches / max(1, len(retrieved_chunks))

    
    rule_checks: List[Dict[str, Any]] = []
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        rules = []

    for rule in rules:
        rule_checks.append({
            "rule": rule,
            "passed": rule.lower() in summary.lower(),
            "comment": "Rule phrase found in summary."
                       if rule.lower() in summary.lower()
                       else "Rule phrase not found in summary."
        })

    
    overall_passed = (
        relevance_score >= min_relevance and
        word_count >= min_word_count
    )

    return ValidationReport(
        query=query,
        summary=summary,
        retrieved_chunks=retrieved_chunks,
        relevance_score=relevance_score,
        word_count=word_count,
        overall_passed=overall_passed,
        rule_checks=rule_checks,
    )
def pretty_print_report(report: ValidationReport) -> str:
    """Return a nicely formatted string version of the validation report."""
    lines = []

    lines.append(f"Query: {report.query}")
    lines.append(f"Summary: {report.summary}")
    lines.append(f"Word count: {report.word_count}")
    lines.append(f"Relevance score (0–1): {report.relevance_score:.2f}")
    lines.append(f"Overall passed: {report.overall_passed}")
    lines.append("")
    lines.append("--- Rules ---")

    if not report.rule_checks:
        lines.append("No explicit rules file found. Manual review required.")
    else:
        for rc in report.rule_checks:
            status = "PASS" if rc["passed"] else "FAIL"
            lines.append(f"{status}  Rule: {rc['rule']}")
            lines.append(f"     Comment: {rc['comment']}")

    return "\n".join(lines)
