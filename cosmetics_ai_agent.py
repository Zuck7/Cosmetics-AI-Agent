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

from rag.cosing_loader import load_cosing

# --- PDF Loader ---
def load_pdfs(folder_path):
    try:
        import PyPDF2
    except ImportError:
        print("PyPDF2 not installed. Using mock cosmetics data for demo.")
        return _get_mock_cosmetics_data()
    folder = Path(folder_path)
    documents = []
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {folder_path}. Using mock cosmetics data for demo.")
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
            print(f"Error reading {pdf_path.name}: {e}. Skipping.")
    if not documents:
        print(f"Could not load any PDFs. Using mock cosmetics data for demo.")
        return _get_mock_cosmetics_data()
    return documents

def _get_mock_cosmetics_data():
    return [
        {"file": "mock_cosmetics_ingredients.txt", "text": """
Retinol is a form of vitamin A that is commonly used in skincare products. It works by increasing cell turnover and stimulating collagen production, which helps reduce the appearance of fine lines, wrinkles, and acne scars. Retinol is considered one of the most effective anti-aging ingredients available. It can also help improve skin texture and tone, reduce hyperpigmentation, and unclog pores. However, retinol can cause irritation, especially when first starting to use it, so it's important to start with a low concentration and gradually increase usage.

Hyaluronic Acid is a naturally occurring substance in the human body that can hold up to 1000 times its weight in water. In skincare, hyaluronic acid is prized for its incredible hydrating properties. It acts as a humectant, drawing moisture from the environment into the skin and helping to maintain hydration levels. This ingredient is suitable for all skin types, including oily and acne-prone skin, as it provides hydration without clogging pores. Hyaluronic acid helps plump the skin, reducing the appearance of fine lines and giving skin a more youthful, dewy appearance.

Vitamin C is a powerful antioxidant that offers multiple benefits for the skin. It helps protect against environmental damage from UV rays and pollution, brightens the complexion, and can help reduce dark spots and hyperpigmentation. Vitamin C also stimulates collagen production, which helps maintain skin firmness and elasticity. It's often used in serums and can help achieve a more radiant, even-toned complexion. However, vitamin C can be unstable and may cause irritation in some people, especially those with sensitive skin.

Niacinamide, also known as vitamin B3, is a versatile ingredient that offers multiple benefits for the skin. It helps regulate oil production, making it excellent for those with oily or acne-prone skin. Niacinamide also helps strengthen the skin barrier, improve skin texture, and reduce the appearance of enlarged pores. Additionally, it can help reduce inflammation and redness, making it suitable for sensitive skin types. This ingredient also has brightening properties and can help reduce the appearance of dark spots and uneven skin tone.

Peptides are short chains of amino acids that serve as building blocks for proteins like collagen and elastin in the skin. In skincare, peptides can help signal the skin to produce more collagen, which can improve skin firmness and reduce the appearance of fine lines and wrinkles. Different types of peptides offer various benefits, including improved skin barrier function, enhanced healing, and better overall skin texture. Peptides are generally well-tolerated by most skin types and are often found in anti-aging serums and moisturizers.
        """},
        {"file": "mock_cosmetics_formulations.txt", "text": """
Foundation is a base makeup product designed to even out skin tone and create a smooth canvas for other makeup products. Modern foundations come in various formulations including liquid, powder, cream, and stick forms. Liquid foundations are the most popular and can range from light coverage for a natural look to full coverage for special occasions. When choosing a foundation, it's important to match your skin tone and consider your skin type - oily skin may benefit from oil-free or matte formulations, while dry skin may need hydrating or dewy finishes.

Serums are concentrated skincare treatments that contain high levels of active ingredients. They are typically thinner in consistency than moisturizers and are designed to penetrate deeper into the skin. Serums can target specific skin concerns such as aging, hyperpigmentation, hydration, or acne. Popular serum ingredients include vitamin C for brightening, retinol for anti-aging, hyaluronic acid for hydration, and niacinamide for oil control. Serums are usually applied after cleansing but before moisturizing, and a little goes a long way.

Face Masks are intensive skincare treatments that can provide targeted benefits for specific skin concerns. Clay masks are excellent for oily and acne-prone skin as they help absorb excess oil and unclog pores. Sheet masks are convenient and often contain hydrating or brightening ingredients. Overnight masks provide extended treatment while you sleep. Exfoliating masks help remove dead skin cells and improve skin texture. Most face masks should be used 1-3 times per week, depending on the type and your skin's needs.

Moisturizers are essential skincare products that help maintain the skin's hydration and strengthen the skin barrier. They come in various formulations to suit different skin types - gel moisturizers for oily skin, cream moisturizers for dry skin, and lotion moisturizers for normal to combination skin. Good moisturizers contain ingredients like ceramides, hyaluronic acid, and glycerin to help lock in moisture. Some moisturizers also include additional beneficial ingredients like SPF for sun protection or anti-aging compounds like retinol or peptides.
        """},
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
    """Simple word-based embedding that captures basic semantic similarity."""
    # Create a simple vocabulary-based embedding
    # This is much better than random embeddings for demo purposes
    words = text.lower().split()

    # Define some cosmetics-related word categories with weights
    skincare_words = {
        'retinol': 1.0, 'vitamin': 0.8, 'acid': 0.7, 'skin': 0.6, 'aging': 0.9,
        'hyaluronic': 1.0, 'hydration': 0.8, 'moisture': 0.7, 'water': 0.5,
        'peptides': 1.0, 'collagen': 0.9, 'elastin': 0.8, 'amino': 0.7,
        'niacinamide': 1.0, 'oil': 0.6, 'pores': 0.7, 'inflammation': 0.8,
        'serum': 0.8, 'moisturizer': 0.8, 'cleanser': 0.7, 'treatment': 0.6,
        'foundation': 1.0, 'makeup': 0.8, 'coverage': 0.7, 'tone': 0.6,
        'mask': 0.8, 'clay': 0.7, 'exfoliating': 0.8, 'sheet': 0.6,
        'wrinkles': 0.9, 'lines': 0.8, 'texture': 0.7, 'brightening': 0.8,
        'antioxidant': 0.9, 'protection': 0.7, 'barrier': 0.8, 'sensitive': 0.6
    }

    # Create embedding vector (384 dimensions to match typical sentence transformers)
    embedding = np.zeros(384, dtype=np.float32)

    # Fill embedding based on word presence and importance
    for i, word in enumerate(words[:384]):  # Use first 384 words max
        # Base position encoding
        embedding[i % 384] += 0.1

        # Add semantic weights for recognized words
        if word in skincare_words:
            # Distribute the word's importance across multiple dimensions
            weight = skincare_words[word]
            for j in range(min(10, 384 - i)):  # Spread across 10 dimensions
                if i + j < 384:
                    embedding[i + j] += weight * 0.1

    # Add some word-specific features
    for word in words:
        if word in skincare_words:
            # Use hash to determine which dimensions to activate
            import hashlib
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for k in range(5):  # Activate 5 dimensions per word
                dim = (word_hash + k) % 384
                embedding[dim] += skincare_words[word] * 0.2

    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    else:
        # If all zeros, create minimal random embedding
        embedding = np.random.randn(384).astype(np.float32) * 0.01
        embedding = embedding / np.linalg.norm(embedding)

    return embedding

class VectorStore:
    def __init__(self, embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        if _HAVE_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer(embed_model)
            self._embed_fn = lambda x: self.model.encode(x)
        else:
            print("sentence-transformers not installed. Using lightweight embedding fallback.")
            self.model = None
            self._embed_fn = lambda texts: np.array([_get_simple_embedding(t) for t in (texts if isinstance(texts, list) else [texts])]).astype(np.float32)
        self.index = None
        self.embeddings = None
        self.text_chunks = []
        self._use_faiss = _HAVE_FAISS
    def chunk_text(self, text, chunk_size=300):
        import re

        # First, try to split by paragraphs (double newlines) for better semantic chunks
        paragraphs = re.split(r'\n\s*\n', text.strip())

        chunks = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If paragraph is small enough, use it as a single chunk
            if len(paragraph.split()) <= chunk_size:
                if len(paragraph.split()) >= 20:  # Only include substantial chunks
                    chunks.append(paragraph)
                continue

            # If paragraph is too large, split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
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
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= 20:  # Only include substantial chunks
                    chunks.append(chunk_text)

        # If no chunks were created, fall back to simple word-based chunking
        if not chunks:
            words = text.split()
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
            chunks = [c for c in chunks if len(c.split()) >= 20]

        return chunks
    def build(self, documents):
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc["text"])
            for c in chunks:
                all_chunks.append(c)
        self.text_chunks = all_chunks
        if not all_chunks:
            print("No text chunks to build index from.")
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
            print("FAISS not installed. Using cosine similarity search (slower).")
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
        # Enhanced fallback: extract sentences most relevant to the question
        import re

        # Try to split into question and context
        if text.lower().startswith("question:") and "context:" in text.lower():
            parts = text.split("Context:", 1) if "Context:" in text else text.split("context:", 1)
            if len(parts) == 2:
                question_part = parts[0].replace("Question:", "").replace("question:", "").strip()
                context_part = parts[1].strip()
            else:
                question_part = ""
                context_part = text
        else:
            question_part = ""
            context_part = text

        sentences = re.split(r'(?<=[.!?])\s+', context_part)

        if question_part:
            # Score sentences by overlap with query words
            query_words = set(question_part.lower().split())
            scored = []
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                # Score based on word overlap and sentence length
                s_words = set(s.lower().split())
                overlap = len(query_words.intersection(s_words))
                # Prefer sentences with good overlap and reasonable length
                score = overlap * 10 + min(len(s.split()), 20)  # Bonus for reasonable length
                if overlap > 0:
                    scored.append((score, s))

            # Sort by score, take top sentences up to max_len words
            summary = []
            word_count = 0
            for _, s in sorted(scored, reverse=True):
                if word_count + len(s.split()) > max_len:
                    break
                summary.append(s)
                word_count += len(s.split())

            if summary:
                return ' '.join(summary)

        # Fallback: if no question or no good matches, take first sentences
        summary = []
        word_count = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
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
    def __init__(self, data_path="data/cosmetics", cosing_csv_path="data/cosing/COSING_Ingredients-Fragrance.Inventory_v2.csv"):
        docs = load_pdfs(data_path) + load_cosing(cosing_csv_path)
        store = VectorStore()
        store.build(docs)
        rag = RAGSystem(store)
        summarizer = SummarizationAgent()
        validator = ValidationAgent()
        reflector = ReflectiveAgent()
        self.planner = PlannerAgent(rag_system=rag, summarizer=summarizer, validator=validator, reflector=reflector)
    def ask(self, query: str, retrieval_k: int = 4, max_summary_len: int = 200, auto_validate: bool = False, auto_reflect: bool = False) -> Dict[str, Any]:
        return self.planner.coordinate(query, retrieval_k=retrieval_k, max_summary_len=max_summary_len, auto_validate=auto_validate, auto_reflect=auto_reflect)
