"""
Unified Cosmetics AI Agent

This file contains all agent classes and orchestration logic for the Cosmetics QA system:
- PlannerAgent
- SummarizationAgent
- RAGSystem
- ValidationAgent (optional)
- ReflectiveAgent (optional)

Usage:
    from cosmetics_ai_agent import CosmeticsAIAgent
    agent = CosmeticsAIAgent(data_path="data/cosmetics")
    result = agent.ask("What is hyaluronic acid?")
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# --- PDF Loader ---
def load_pdfs(folder_path):
    try:
        import PyPDF2
    except ImportError:
        print("⚠️  PyPDF2 not installed. Using mock cosmetics data for demo.")
        return _get_mock_cosmetics_data()
    folder = Path(folder_path)
    documents = []
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDFs found in {folder_path}. Using mock cosmetics data for demo.")
        return _get_mock_cosmetics_data()
    for pdf_path in pdf_files:
        text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            documents.append({"file": pdf_path.name, "text": text})
        except Exception as e:
            print(f"⚠️  Error reading {pdf_path.name}: {e}. Skipping.")
    if not documents:
        print(f"⚠️  Could not load any PDFs. Using mock cosmetics data for demo.")
        return _get_mock_cosmetics_data()
    return documents

def _get_mock_cosmetics_data():
    return [
        {"file": "mock_cosmetics_ingredients.txt", "text": "Retinol: ... Hyaluronic Acid: ... Vitamin C: ... Peptides: ... Niacinamide: ..."},
        {"file": "mock_cosmetics_formulations.txt", "text": "Foundation: ... Serums: ... Face Masks: ... Moisturizers: ..."},
    ]

# --- Vector Store ---
try:
    import faiss
    _HAVE_FAISS = True
except ImportError:
    _HAVE_FAISS = False
try:
    from sentence_transformers import SentenceTransformer
    _HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAVE_SENTENCE_TRANSFORMERS = False
import numpy as np

def _cosine_similarity(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_normalized = a / (a_norm + 1e-8)
    b_normalized = b / (b_norm + 1e-8)
    return np.dot(a_normalized, b_normalized.T)

def _get_simple_embedding(text):
    import hashlib
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    np.random.seed(seed % 2**31)
    embedding = np.random.randn(384).astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

class VectorStore:
    def __init__(self, embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        if _HAVE_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer(embed_model)
            self._embed_fn = lambda x: self.model.encode(x)
        else:
            print("⚠️  sentence-transformers not installed. Using lightweight embedding fallback.")
            self.model = None
            self._embed_fn = lambda texts: np.array([_get_simple_embedding(t) for t in (texts if isinstance(texts, list) else [texts])]).astype(np.float32)
        self.index = None
        self.embeddings = None
        self.text_chunks = []
        self._use_faiss = _HAVE_FAISS
    def chunk_text(self, text, chunk_size=300):
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            words = len(sentence.split())
            if current_length + words > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = words
            else:
                current_chunk.append(sentence)
                current_length += words
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        chunks = [c for c in chunks if len(c.split()) >= 20]
        if not chunks:
            words = text.split()
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    def build(self, documents):
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc["text"])
            for c in chunks:
                all_chunks.append(c)
        self.text_chunks = all_chunks
        if not all_chunks:
            print("⚠️  No text chunks to build index from.")
            return
        embeddings = self._embed_fn(all_chunks)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = embeddings.astype(np.float32)
        if self._use_faiss:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            self.index = index
        else:
            print("⚠️  FAISS not installed. Using cosine similarity search (slower).")
            self.embeddings = embeddings
    def search(self, query, k=5):
        if not self.text_chunks:
            return []
        query_emb = self._embed_fn([query]).astype(np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        if self._use_faiss and self.index is not None:
            distances, idxs = self.index.search(query_emb, min(k, len(self.text_chunks)))
            return [self.text_chunks[i] for i in idxs[0] if i < len(self.text_chunks)]
        else:
            if self.embeddings is None:
                return []
            similarities = _cosine_similarity(query_emb, self.embeddings)[0]
            top_idxs = np.argsort(-similarities)[:min(k, len(self.text_chunks))]
            return [self.text_chunks[i] for i in top_idxs]

# --- RAG System ---
class RAGSystem:
    def __init__(self, store):
        self.store = store
    def query(self, question, k=5):
        results = self.store.search(question, k=max(k * 2, 10))
        filtered_results = [r for r in results if len(r.split()) >= 30]
        if len(filtered_results) < k:
            filtered_results = results
        final_results = filtered_results[:k]
        context = "\n\n".join(final_results)
        response = {
            "question": question,
            "context": context,
            "chunks": final_results
        }
        return response

# --- Summarization Agent ---
class SummarizationAgent:
    def summarize(self, text, max_len=200):
        # Fallback: extract first N sentences up to max_len words
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        summary = []
        word_count = 0
        for s in sentences:
            if word_count + len(s.split()) > max_len:
                break
            summary.append(s)
            word_count += len(s.split())
        return ' '.join(summary)

# --- Validation Agent (optional stub) ---
class ValidationAgent:
    def validate(self, summary: str, sources: List[str], query: str) -> Dict[str, Any]:
        # Stub: always valid
        return {"is_valid": True, "confidence": 1.0, "issues": []}

# --- Reflective Agent (optional stub) ---
class ReflectiveAgent:
    def reflect(self, answer: str, query: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        # Stub: no suggestions
        return {"needs_improvement": False, "suggestions": []}

# --- Planner Agent ---
class PlannerAgent:
    def __init__(self, rag_system, summarizer, validator=None, reflector=None):
        self.rag_system = rag_system
        self.summarizer = summarizer
        self.validator = validator
        self.reflector = reflector
    def coordinate(self, user_query: str, retrieval_k: int = 4, max_summary_len: int = 200, auto_validate: bool = False, auto_reflect: bool = False) -> Dict[str, Any]:
        rag_output = self.rag_system.query(user_query, k=retrieval_k)
        summary = self.summarizer.summarize(f"Question: {user_query}\nContext: {rag_output['context']}", max_len=max_summary_len)
        validation = self.validator.validate(summary, rag_output.get("chunks", []), user_query) if (auto_validate and self.validator) else {"is_valid": True, "confidence": 1.0, "issues": []}
        reflection = self.reflector.reflect(summary, user_query, validation) if (auto_reflect and self.reflector) else None
        final_response = {
            "query": user_query,
            "answer": summary,
            "supporting_evidence": rag_output.get("chunks", []),
            "validation": validation,
            "metadata": {
                "retrieved_chunks": len(rag_output.get("chunks", [])),
                "needs_improvement": reflection.get("needs_improvement", False) if reflection else False
            }
        }
        if reflection and reflection.get("suggestions"):
            final_response["improvement_suggestions"] = reflection["suggestions"]
        return final_response

# --- Unified Agent Interface ---
class CosmeticsAIAgent:
    def __init__(self, data_path="data/cosmetics"):
        docs = load_pdfs(data_path)
        store = VectorStore()
        store.build(docs)
        rag = RAGSystem(store)
        summarizer = SummarizationAgent()
        validator = ValidationAgent()
        reflector = ReflectiveAgent()
        self.planner = PlannerAgent(rag_system=rag, summarizer=summarizer, validator=validator, reflector=reflector)
    def ask(self, query: str, retrieval_k: int = 4, max_summary_len: int = 200, auto_validate: bool = False, auto_reflect: bool = False) -> Dict[str, Any]:
        return self.planner.coordinate(query, retrieval_k=retrieval_k, max_summary_len=max_summary_len, auto_validate=auto_validate, auto_reflect=auto_reflect)
