from planner_agent.planner import PlannerAgent
from summarization_agent.sum_agent import SummarizationAgent


class DummyRAG:
    def query(self, question, k=4):
        return {
            "question": question,
            "context": "Retinol and vitamin C are widely used for anti-aging skincare.",
            "chunks": [
                "Retinol supports cell turnover.",
                "Vitamin C can support collagen and brightening.",
            ],
        }


class FailingCompletions:
    def create(self, **kwargs):
        raise Exception("401 invalid_api_key")


class FailingChat:
    completions = FailingCompletions()


class FailingClient:
    chat = FailingChat()


class BrokenSummarizer:
    def summarize(self, text, max_len=200):
        raise RuntimeError("summarizer unavailable")


def test_summarizer_falls_back_when_groq_call_fails():
    agent = SummarizationAgent()
    agent.client = FailingClient()

    text = "Question: What does retinol do?\nContext: Retinol improves skin texture and fine lines."
    summary = agent.summarize(text, max_len=40)

    assert isinstance(summary, str)
    assert len(summary.strip()) > 0


def test_planner_coordinate_returns_expected_shape_in_demo_mode():
    planner = PlannerAgent(rag_system=DummyRAG(), summarizer=SummarizationAgent())

    result = planner.coordinate("What helps with anti aging?", retrieval_k=2)

    assert isinstance(result, dict)
    assert result["query"] == "What helps with anti aging?"
    assert isinstance(result.get("answer"), str)
    assert len(result["answer"].strip()) > 0
    assert isinstance(result.get("supporting_evidence"), list)
    assert "validation" in result
    assert "metadata" in result


def test_planner_uses_direct_fallback_when_summarizer_fails():
    planner = PlannerAgent(rag_system=DummyRAG(), summarizer=BrokenSummarizer())

    result = planner.coordinate("What helps with anti aging?", retrieval_k=2)

    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0
