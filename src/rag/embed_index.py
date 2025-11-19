# =========================================================
# src/rag/embed_index.py — Create FAISS index from JSONL chunks
# =========================================================
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

class EmbedIndexer:
    def __init__(self, kb_dir: str = "../../data/knowledge_base",
                 embed_model: str = "all-MiniLM-L6-v2",
                 index_file: str = "faiss.index",
                 chunks_file: str = "chunks.json"):
        self.kb_dir = Path(kb_dir).resolve()
        self.chunks_file = self.kb_dir / chunks_file
        self.index_file = self.kb_dir / index_file
        self.embed_model = embed_model
        self.model = None
        self.chunks = []
        self.embeddings = None

    def load_chunks(self):
        print("\nLoading chunks from:", self.chunks_file)
        if not self.chunks_file.exists():
            raise RuntimeError(f"❌ chunks.json not found at {self.chunks_file}")

        # Read JSONL
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            self.chunks = [json.loads(line)["chunk"] for line in f if line.strip()]

        if not self.chunks:
            raise RuntimeError("❌ No chunks found in chunks.json")
        print(f"Total chunks loaded: {len(self.chunks)}")

    def load_model(self):
        print("\nLoading embedding model:", self.embed_model)
        self.model = SentenceTransformer(self.embed_model)

    def create_embeddings(self):
        print("\nGenerating embeddings (CPU-friendly)...")
        self.embeddings = self.model.encode(
            self.chunks, batch_size=16, show_progress_bar=True
        )
        self.embeddings = np.array(self.embeddings, dtype="float32")
        print("Embeddings shape:", self.embeddings.shape)

    def build_faiss_index(self):
        dim = self.embeddings.shape[1]
        print("\nCreating FAISS index with dimension:", dim)
        index = faiss.IndexFlatL2(dim)
        index.add(self.embeddings)
        print("Total vectors in FAISS:", index.ntotal)

        # Save FAISS index (convert Path to str)
        faiss.write_index(index, str(self.index_file))
        print(f"\n✔ FAISS index saved to: {self.index_file}")

    def run(self):
        self.load_chunks()
        self.load_model()
        self.create_embeddings()
        self.build_faiss_index()
        print("\n🎉 Embedding + Index creation completed successfully!")


if __name__ == "__main__":
    indexer = EmbedIndexer()
    indexer.run()
