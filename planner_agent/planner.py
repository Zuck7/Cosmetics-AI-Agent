import json
from typing import Dict, Any, List, Optional
from groq_config import get_groq_client, get_groq_api_key

# Optional adapters for validation and reflection
try:
    from validation_agent.validation_agent import validate_summary, ValidationReport
except Exception:
    validate_summary = None
    ValidationReport = None

try:
    from reflective_agent.reflective_agent import reflect_on_run
except Exception:
    reflect_on_run = None


class PlannerAgent:
    """
    Top-level coordinator agent that orchestrates the entire multi-agent system.
    Responsible for:
    - Interpreting user queries
    - Task decomposition
    - Delegating tasks to specialized worker agents
    - Aggregating results into structured responses
    """
    
    def __init__(self, rag_system, summarizer, model_name="llama-3.3-70b-versatile", groq_api_key=None):
        """
        Initialize the Planner Agent with worker agents.
        
        Args:
            rag_system: RAGSystem instance for document retrieval
            summarizer: SummarizationAgent instance for text condensation
            model_name: LLM model for planning and orchestration
            groq_api_key: Deprecated. GROQ_API_KEY from .env is used globally.
        
        """
        self.model_name = model_name
        # Keep parameter for backward compatibility, but enforce one project key source.
        if groq_api_key is not None:
            print("PlannerAgent ignores custom groq_api_key and uses GROQ_API_KEY from .env.")

        self.client = get_groq_client()
        if self.client is None:
            if get_groq_api_key() is None:
                print("GROQ_API_KEY not found. PlannerAgent will use demo mode.")
            else:
                print("Groq client unavailable. PlannerAgent will use demo mode.")
        
        # Worker agents
        self.rag_system = rag_system
        self.summarizer = summarizer
        # Validator and Reflector agent integration
        # If validation and reflection modules are available, create small adapters
        if validate_summary and ValidationReport:
            class _ValidatorAdapter:
                def validate(self, summary: str, sources: List[str], query: str) -> Dict[str, Any]:
                    # validate_summary returns a ValidationReport dataclass
                    report = validate_summary(query=query, summary=summary, retrieved_chunks=sources)
                    issues = []
                    for rc in report.rule_checks:
                        if not rc.get("passed", False):
                            issues.append(f"Rule failed: {rc.get('rule')} - {rc.get('comment')}")

                    return {
                        "is_valid": bool(report.overall_passed),
                        "confidence": float(report.relevance_score),
                        "issues": issues,
                        "_report": report,
                    }

            self.validator = _ValidatorAdapter()
        else:
            self.validator = None

        if reflect_on_run and ValidationReport:
            class _ReflectorAdapter:
                def reflect(self, answer: str, query: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
                    # The reflector expects a ValidationReport. Try to extract it.
                    report = None
                    if isinstance(validation_result, dict) and "_report" in validation_result:
                        report = validation_result["_report"]
                    # If no ValidationReport available, create a minimal one if possible
                    if report is None and ValidationReport is not None:
                        try:
                            report = ValidationReport(query=query, summary=answer, retrieved_chunks=validation_result.get("retrieved_chunks", []), relevance_score=validation_result.get("confidence", 0.0), word_count=len(answer.split()), overall_passed=validation_result.get("is_valid", True), rule_checks=[])
                        except Exception:
                            report = None

                    if report is None:
                        # Fallback: call reflector with None-safe input
                        return reflect_on_run(validation_result) if callable(reflect_on_run) else {"needs_improvement": False, "suggestions": []}

                    return reflect_on_run(report)

            self.reflector = _ReflectorAdapter()
        else:
            self.reflector = None
        
    def decompose_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze the user query and determine execution plan.
        
        Args:
            user_query: User's research question
            
        Returns:
            Dictionary containing task decomposition plan
        """
        if self.client is None:
            # Demo mode: return default plan
            return {
                "needs_retrieval": True,
                "retrieval_k": 4,
                "needs_summarization": True,
                "needs_validation": False,
                "complexity": "moderate",
                "query_intent": "Query about cosmetics"
            }
        
        prompt = f"""You are a planner agent for a cosmetics research system. Analyze this query and create an execution plan.

User Query: "{user_query}"

Determine:
1. Does this require document retrieval? (yes/no)
2. How many relevant chunks should be retrieved? (1-10)
3. Does this require summarization? (yes/no)
4. Does this require validation for accuracy? (yes/no)
5. What is the query complexity? (simple/moderate/complex)

Respond in JSON format:
{{
    "needs_retrieval": true/false,
    "retrieval_k": number,
    "needs_summarization": true/false,
    "needs_validation": true/false,
    "complexity": "simple|moderate|complex",
    "query_intent": "brief description of what user wants"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            return plan
        except Exception as e:
            print(f"⚠️  Query decomposition failed: {e}. Using default plan.")
            return {
                "needs_retrieval": True,
                "retrieval_k": 4,
                "needs_summarization": True,
                "needs_validation": False,
                "complexity": "moderate",
                "query_intent": "Query about cosmetics"
            }
    
    def execute_retrieval(self, query: str, k: int = 4) -> Dict[str, Any]:
        """
        Execute retrieval task using RAG system.
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            
        Returns:
            RAG system output with context and chunks
        """
        print(f"[PLANNER] Executing retrieval (k={k})...")
        return self.rag_system.query(query, k=k)
    
    def execute_summarization(self, text: str, query: str = "", max_len: int = 200) -> str:
        """
        Execute summarization task.
        
        Args:
            text: Text to summarize
            query: User query for context-aware summarization
            max_len: Maximum summary length in words
            
        Returns:
            Summarized text
        """
        print(f"[PLANNER] Executing summarization (max_len={max_len})...")
        # Pass query to summarizer if supported, otherwise concatenate
        if hasattr(self.summarizer, "summarize"):
            try:
                return self.summarizer.summarize(f"Question: {query}\nContext: {text}", max_len=max_len)
            except Exception:
                try:
                    return self.summarizer.summarize(text, max_len=max_len)
                except Exception as e:
                    print(f"[PLANNER] Summarization failed ({e}). Using direct-answer fallback.")
                    return self._generate_direct_answer(query, text)
        else:
            return text[:max_len*5]  # fallback: truncate
    
    def execute_validation(self, summary: str, sources: List[str], query: str) -> Dict[str, Any]:
        """
        Execute validation task to check for hallucinations and inconsistencies.
        
        Args:
            summary: Generated summary to validate
            sources: Source chunks used for the summary
            query: Original user query
            
        Returns:
            Validation results
        """
        if not self.validator:
            print("[PLANNER] Validation agent not available, skipping...")
            return {"is_valid": True, "confidence": 1.0, "issues": []}
        
        print("[PLANNER] Executing validation...")
        return self.validator.validate(summary, sources, query)
    
    def execute_reflection(self, answer: str, query: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute reflection task for self-critique and improvement suggestions.
        
        Args:
            answer: Generated answer
            query: Original user query
            validation_result: Results from validation
            
        Returns:
            Reflection results with improvement suggestions
        """
        if not self.reflector:
            print("[PLANNER] Reflective agent not available, skipping...")
            return {"needs_improvement": False, "suggestions": []}
        
        print("[PLANNER] Executing reflection...")
        return self.reflector.reflect(answer, query, validation_result)
    
    def merge_results(self, query: str, rag_output: Dict[str, Any], summary: str, 
                     validation: Dict[str, Any], reflection: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Merge all worker outputs into a structured final response.
        
        Args:
            query: Original user query
            rag_output: Output from RAG system
            summary: Summarized answer
            validation: Validation results
            reflection: Reflection results (optional)
            
        Returns:
            Structured final response
        """
        print("[PLANNER] Merging results into final response...")
        
        final_response = {
            "query": query,
            "answer": summary,
            "supporting_evidence": rag_output.get("chunks", []),
            "validation": {
                "is_valid": validation.get("is_valid", True),
                "confidence": validation.get("confidence", 1.0),
                "issues": validation.get("issues", [])
            },
            "metadata": {
                "retrieved_chunks": len(rag_output.get("chunks", [])),
                "needs_improvement": reflection.get("needs_improvement", False) if reflection else False
            }
        }
        
        if reflection and reflection.get("suggestions"):
            final_response["improvement_suggestions"] = reflection["suggestions"]
        
        return final_response
    
    def coordinate(self, user_query: str, auto_validate: bool = False, auto_reflect: bool = False, retrieval_k: Optional[int] = None, max_summary_len: Optional[int] = None) -> Dict[str, Any]:
        """
        Main coordination method that orchestrates the entire workflow.
        
        Args:
            user_query: User's research question
            auto_validate: Whether to automatically run validation (requires validator)
            auto_reflect: Whether to automatically run reflection (requires reflector)
            
        Returns:
            Final structured response
        
        Note: Validation and reflection features will be enabled once agent integration is complete.
        """
        print(f"\n{'='*60}")
        print(f"[PLANNER] Starting coordination for query: {user_query}")
        print(f"{'='*60}\n")
        
        # Step 1: Decompose query and create execution plan
        plan = self.decompose_query(user_query)
        print(f"[PLANNER] Execution plan: {json.dumps(plan, indent=2)}\n")
        
        # Step 2: Execute retrieval
        rag_output = None
        if plan.get("needs_retrieval", True):
            k = plan.get("retrieval_k", 4)
            if retrieval_k is not None:
                k = retrieval_k
            rag_output = self.execute_retrieval(user_query, k=k)
        else:
            print("[PLANNER] Skipping retrieval (not needed)...")
            rag_output = {"question": user_query, "context": "", "chunks": []}
        
        # Step 3: Execute summarization
        summary = ""
        if plan.get("needs_summarization", True) and rag_output["context"]:
            max_len = None
            if max_summary_len is not None:
                max_len = max_summary_len
            summary = self.execute_summarization(rag_output["context"], user_query, max_len=max_len or 200)
        else:
            # If no summarization needed, use raw context or generate direct answer
            if rag_output["context"]:
                summary = self._generate_direct_answer(user_query, rag_output["context"])
            else:
                summary = "No relevant information found in the knowledge base."
        
        # Step 4: Execute validation
        validation = {"is_valid": True, "confidence": 1.0, "issues": []}
        if auto_validate and plan.get("needs_validation", True):
            validation = self.execute_validation(summary, rag_output.get("chunks", []), user_query)
        
        # Step 5: Execute reflection
        reflection = None
        if auto_reflect or (validation and not validation.get("is_valid", True)):
            reflection = self.execute_reflection(summary, user_query, validation)
            
            # If reflection suggests improvements, optionally refine the answer
            if reflection.get("needs_improvement"):
                print("[PLANNER] Reflection suggests improvements. Consider refining...")
        
        # Step 6: Merge and return final response
        final_response = self.merge_results(user_query, rag_output, summary, validation, reflection)
        
        print(f"\n{'='*60}")
        print("[PLANNER] Coordination complete")
        print(f"{'='*60}\n")
        
        return final_response
    
    def _generate_direct_answer(self, query: str, context: str) -> str:
        """
        Generate a direct answer without summarization when appropriate.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Direct answer
        """
        print("[PLANNER] Generating direct answer...")
        
        if self.client is None:
            # Demo mode: extract sentences most related to the query
            import re
            sentences = re.split(r'(?<=[.!?])\s+', context)
            query_words = set(query.lower().split())
            relevant = [s for s in sentences if any(w in s.lower() for w in query_words)]
            if relevant:
                return ' '.join(relevant[:3])
            # fallback: first 2 sentences
            return ' '.join(sentences[:2]) if sentences else context[:300]

        prompt = f"""Based on the following context, answer the user's question directly and concisely.

Context:
{context}

Question: {query}

Provide a clear, evidence-based answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Direct answer generation failed: {e}. Using context excerpt.")
            lines = context.split('\n')
            relevant = [l for l in lines if any(w in l.lower() for w in query.lower().split())][:2]
            return '\n'.join(relevant) if relevant else context[:300]
    
    def format_output(self, response: Dict[str, Any]) -> str:
        """
        Format the final response for user-friendly display.
        
        Args:
            response: Structured response dictionary
            
        Returns:
            Formatted string output
        """
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"QUERY: {response['query']}")
        output.append(f"{'='*60}\n")
        
        output.append("ANSWER:")
        output.append(response['answer'])
        output.append("")
        
        if response['validation']['issues']:
            output.append("VALIDATION ISSUES:")
            for issue in response['validation']['issues']:
                output.append(f"  - {issue}")
            output.append("")
        
        if response.get('improvement_suggestions'):
            output.append("IMPROVEMENT SUGGESTIONS:")
            for suggestion in response['improvement_suggestions']:
                output.append(f"  - {suggestion}")
            output.append("")
        
        output.append(f"\n METADATA:")
        output.append(f"  - Retrieved chunks: {response['metadata']['retrieved_chunks']}")
        output.append(f"  - Validation confidence: {response['validation']['confidence']:.2f}")
        output.append(f"  - Status: {'Valid' if response['validation']['is_valid'] else 'Needs Review'}")
        
        return "\n".join(output)
