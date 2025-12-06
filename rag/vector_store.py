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
    """Lightweight fallback: deterministic embedding using word hash."""
    import hashlib
    words = text.lower().split()
    # Use text hash as seed for reproducibility
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    np.random.seed(seed % 2**31)
    embedding = np.random.randn(384).astype(np.float32)
    
    # Normalize
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
        
        # Filter out very short chunks (less than 20 words) that might be headers
        chunks = [c for c in chunks if len(c.split()) >= 20]
        
        # If all chunks were filtered, fall back to original simple chunking
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
            # Fallback: store embeddings for cosine similarity search
            print("⚠️  FAISS not installed. Using cosine similarity search (slower).")
            self.embeddings = embeddings

    def search(self, query, k=5):
        if not self.text_chunks:
            return []

        query_emb = self._embed_fn([query]).astype(np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        
        if self._use_faiss and self.index is not None:
            # Use FAISS for fast search
            distances, idxs = self.index.search(query_emb, min(k, len(self.text_chunks)))
            return [self.text_chunks[i] for i in idxs[0] if i < len(self.text_chunks)]
        else:
            # Fallback: cosine similarity search
            if self.embeddings is None:
                return []
            
            similarities = _cosine_similarity(query_emb, self.embeddings)[0]
            top_idxs = np.argsort(-similarities)[:min(k, len(self.text_chunks))]
            return [self.text_chunks[i] for i in top_idxs]
