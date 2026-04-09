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
    """Compute cosine similarity between vectors (numpy only)."""
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
            print("⚠️  sentence-transformers not installed. Using lightweight embedding fallback.")
            self.model = None
            self._embed_fn = lambda texts: np.array([_get_simple_embedding(t) for t in (texts if isinstance(texts, list) else [texts])]).astype(np.float32)
        
        self.index = None
        self.embeddings = None
        self.text_chunks = []
        self.chunk_sources = []
        self._use_faiss = _HAVE_FAISS

    def chunk_text(self, text, chunk_size=300):
        """
        Split text into meaningful chunks by sentences, not just word count.
        This preserves semantic meaning and avoids splitting mid-thought.
        """
        # Split by common sentence endings
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Count words in this sentence
            words = len(sentence.split())
            
            # If adding this sentence exceeds chunk_size, save current chunk and start new one
            if current_length + words > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = words
            else:
                current_chunk.append(sentence)
                current_length += words
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Filter out very short chunks (less than 10 words) that might be headers
        chunks = [c for c in chunks if len(c.split()) >= 10]
        
        # If all chunks were filtered, fall back to original simple chunking
        if not chunks:
            words = text.split()
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        return chunks

    def build(self, documents):
        all_chunks = []
        all_sources = []
        for doc in documents:
            chunks = self.chunk_text(doc["text"])
            for c in chunks:
                all_chunks.append(c)
                all_sources.append(doc.get("file", "unknown"))

        self.text_chunks = all_chunks
        self.chunk_sources = all_sources

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
            # Fallback: store embeddings for cosine similarity search
            print("FAISS not installed. Using cosine similarity search (slower).")
            self.embeddings = embeddings

    def _rank_indices(self, query, top_n):
        if not self.text_chunks:
            return []

        query_emb = self._embed_fn([query]).astype(np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        if self._use_faiss and self.index is not None:
            _, idxs = self.index.search(query_emb, min(top_n, len(self.text_chunks)))
            return [i for i in idxs[0] if i < len(self.text_chunks)]

        if self.embeddings is None:
            return []

        similarities = _cosine_similarity(query_emb, self.embeddings)[0]
        top_idxs = np.argsort(-similarities)[:min(top_n, len(self.text_chunks))]
        return list(top_idxs)

    def search(self, query, k=5, return_metadata=False, diversify_sources=True, per_source_cap=2):
        if not self.text_chunks:
            return []

        ranked_indices = self._rank_indices(query, top_n=max(k * 10, 50))
        if not ranked_indices:
            return []

        if not diversify_sources or not self.chunk_sources:
            selected = ranked_indices[:k]
        else:
            selected = []
            selected_set = set()
            source_counts = {}

            # Pass 1: try to include at least one chunk from each source.
            for idx in ranked_indices:
                source = self.chunk_sources[idx]
                if source_counts.get(source, 0) == 0:
                    selected.append(idx)
                    selected_set.add(idx)
                    source_counts[source] = 1
                    if len(selected) >= k:
                        break

            # Pass 2: fill remaining slots while capping per-source chunks.
            if len(selected) < k:
                for idx in ranked_indices:
                    if idx in selected_set:
                        continue
                    source = self.chunk_sources[idx]
                    if source_counts.get(source, 0) < per_source_cap:
                        selected.append(idx)
                        selected_set.add(idx)
                        source_counts[source] = source_counts.get(source, 0) + 1
                        if len(selected) >= k:
                            break

            # Pass 3: if still short, allow any source.
            if len(selected) < k:
                for idx in ranked_indices:
                    if idx in selected_set:
                        continue
                    selected.append(idx)
                    selected_set.add(idx)
                    if len(selected) >= k:
                        break

        if return_metadata:
            return [
                {
                    "text": self.text_chunks[i],
                    "source": self.chunk_sources[i] if i < len(self.chunk_sources) else "unknown",
                }
                for i in selected
            ]

        return [self.text_chunks[i] for i in selected]
