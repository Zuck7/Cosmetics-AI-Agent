
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    def __init__(self, embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)
        self.index = None
        self.text_chunks = []

    def chunk_text(self, text, chunk_size=300):
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def build(self, documents):
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc["text"])
            for c in chunks:
                all_chunks.append(c)

        self.text_chunks = all_chunks

        embeddings = self.model.encode(all_chunks)
        embeddings = np.array(embeddings).astype(np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index

    def search(self, query, k=5):
        query_emb = self.model.encode([query]).astype(np.float32)
        distances, idxs = self.index.search(query_emb, k)
        return [self.text_chunks[i] for i in idxs[0]]
