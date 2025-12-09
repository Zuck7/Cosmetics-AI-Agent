# Cosmetics-AI-Agent

A hierarchical multi-agent system for cosmetics research, featuring intelligent query orchestration, RAG-based retrieval, and evidence-based answer generation.

## Architecture

This system implements a **two-level hierarchical architecture** with a Planner Agent coordinating specialized worker agents:

```
┌─────────────────────────────────────┐
│      PLANNER AGENT (Level 1)       │
│   • Query decomposition             │
│   • Task delegation                 │
│   • Result aggregation              │
└──────────┬──────────────────────────┘
           │
    ┌──────┴────────┬─────────────┬──────────────┐
    ▼               ▼             ▼              ▼
┌────────┐   ┌────────────┐  ┌──────────┐  ┌──────────┐
│  RAG   │   │Summarize   │  │Validate  │  │Reflective│
│ System │   │  Agent     │  │  Agent   │  │  Agent   │
└────────┘   └────────────┘  └──────────┘  └──────────┘
   Level 2 Worker Agents
   
(Carlos)      (Carlos)       (Franklyn)     (Franklyn)
```

## Team Assignments

- **Planner Agent**: Orchestration & coordination (Zuhriddin)
- **RAG System + Summarization Agent**: Carlos
- **Validation Agent + Reflective Agent**: Franklyn

## Features

### Planner Agent
- **Intelligent Query Decomposition**: Analyzes queries and creates optimal execution plans
- **Task Orchestration**: Delegates work to specialized agents
- **Result Aggregation**: Merges outputs into structured, comprehensive responses
- **Evidence Tracking**: Maintains citations and source references

### Worker Agents
1. **RAG System** (Carlos): Document retrieval using vector embeddings
2. **Summarization Agent** (Carlos): Condenses retrieved context into concise answers
3. **Validation Agent** (Franklyn): Checks accuracy and detects hallucinations
4. **Reflective Agent** (Franklyn): Performs self-critique and suggests improvements

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GROQ_API_KEY=your_groq_key_here" > .env
```

### Basic Usage

```python
from planner_agent.planner import PlannerAgent
from rag.pdf_loader import load_pdfs
from rag.vector_store import VectorStore
from rag.rag_system import RAGSystem
from summarization_agent.sum_agent import SummarizationAgent

# Initialize system
docs = load_pdfs("data/cosmetics")
store = VectorStore()
store.build(docs)

rag = RAGSystem(store)
summarizer = SummarizationAgent()

planner = PlannerAgent(
    rag_system=rag,
    summarizer=summarizer
)

# Query the system
query = "What ingredients are commonly used for anti aging skin care?"
result = planner.coordinate(query)

# Display results
print(planner.format_output(result))
```



### Run Examples

```bash
# Run the demo
python agent_usage_sample.py
```

## Project Structure

```
Cosmetics-AI-Agent/
├── agent_usage_sample.py      # Usage examples
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/
│   └── cosmetics/            # PDF knowledge base
│       └── paper_beaty_pred.pdf
│
├── rag/                      # RAG System (Carlos)
│   ├── __init__.py
│   ├── pdf_loader.py        # PDF document loading
│   ├── vector_store.py      # FAISS vector storage
│   └── rag_system.py        # Retrieval logic
│
├── summarization_agent/      # Summarization Agent (Carlos)
│   ├── __init__.py
│   └── sum_agent.py         # Text summarization
│
└── planner_agent/           # Planner Agent
    ├── __init__.py
    └── planner.py          # Orchestration logic
```

## System Workflow

The Planner Agent orchestrates the following workflow:

```
User Query
    ↓
1. Query Decomposition (Planner)
   - Analyze intent
   - Determine execution plan
   - Set retrieval parameters
    ↓
2. Document Retrieval (RAG - Carlos)
   - Semantic search in knowledge base
   - Retrieve top-k relevant chunks
    ↓
3. Summarization (Carlos)
   - Condense retrieved context
   - Generate concise answer
    ↓
4. Validation (Franklyn)
   - Check accuracy against sources
   - Detect hallucinations
    ↓
5. Reflection (Franklyn)
   - Self-critique answer quality
    ↓
6. Result Aggregation (Planner)
   - Merge all outputs
   - Format final response
    ↓
Structured JSON Response
```

## Output Format

The system returns structured responses:

```python
{
    "query": "User's original question",
    "answer": "Summarized, evidence-based answer",
    "supporting_evidence": [
        "Retrieved text chunk 1",
        "Retrieved text chunk 2",
        ...
    ],
    "validation": {
        "is_valid": True,
        "confidence": 1.0,
        "issues": []
    },
    "metadata": {
        "retrieved_chunks": 4
    }
}
```

## Configuration

### Planner Setup

```python
planner = PlannerAgent(
    rag_system=rag,
    summarizer=summarizer,
    model_name="llama-3.1-70b-versatile"  # Customizable model
)
```

The Planner automatically determines optimal retrieval parameters through query decomposition.

## Documentation

- **Usage Examples**: See `agent_usage_sample.py`
- **Design Specifications**: See `Design PART a.docx` and `COMP 248_Project_ Assignment.docx`

## Key Design Principles

1. **Evidence-Based**: All answers grounded in retrieved sources
2. **Two-Level Hierarchy**: Clear separation between coordination and execution
3. **Safety-Sensitive**: Validation ensures accuracy for beauty product information
4. **Transparency**: Confidence scores and source citations provided
5. **Iterative Refinement**: Reflection enables quality improvement

## Technologies

- **LLM Provider**: Groq (Llama 3.1 70B for Planner, Llama 3.1 8B for Summarization)
- **Vector Store**: FAISS
- **Embeddings**: Sentence Transformers
- **PDF Processing**: PyPDF2
- **Framework**: Python 3.8+

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- `groq>=0.37.1`
- `sentence-transformers>=5.1.2`
- `faiss-cpu>=1.13.1`
- `PyPDF2>=3.0.1`
- `python-dotenv>=1.2.1`

## Integration Status

- **Planner Agent**: Complete
- **RAG System**: Complete
- **Summarization Agent**: Complete
- **Validation Agent**: Pending
- **Reflective Agent**: Pending

## License

See LICENSE file for details.

