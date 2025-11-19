import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import threading
import re
from pathlib import Path
from src.rag.config import *


class RAGAssistant:
    def __init__(
        self,
        index_file: str = None,
        chunks_file: str = None,
        embed_model: str = EMBED_MODEL,
        llm_api_key: str = LLM_API_KEY,
        llm_base_url: str = LLM_BASE_URL,
        llm_model: str = LLM_MODEL
    ):
        # Resolve project root (folder that contains 'data')
        project_root = Path(__file__).resolve().parents[2]

        # Correct final paths
        self.index_file = Path(index_file) if index_file else project_root / "data/knowledge_base/faiss.index"
        self.chunks_file = Path(chunks_file) if chunks_file else project_root / "data/knowledge_base/chunks.json"

        self.embed_model_name = embed_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model

        self._index = None
        self._chunks = None
        self._embedder = None
        self._client = None
        self._loaded = False
        self._lock = threading.Lock()

    # -------------------------------
    # Strip Markdown
    # -------------------------------
    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
        text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

    # -------------------------------
    # Load JSONL chunks safely
    # -------------------------------
    @staticmethod
    def _load_chunks_jsonl(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")

        chunks = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    # -------------------------------
    # Lazy load FAISS + Model + LLM
    # -------------------------------
    def _load_resources(self):
        with self._lock:
            if self._loaded:
                return

            print(f"Loading FAISS index from: {self.index_file}")
            self._index = faiss.read_index(str(self.index_file))

            print(f"Loading chunks from: {self.chunks_file}")
            self._chunks = self._load_chunks_jsonl(self.chunks_file)
            print(f"Total chunks: {len(self._chunks)}")

            print(f"Loading embedder: {self.embed_model_name}")
            self._embedder = SentenceTransformer(self.embed_model_name)

            print("Initializing LLM client...")
            if not self.llm_api_key:
                raise ValueError("Missing LLM API key.")
            self._client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)

            self._loaded = True
            print("Resources loaded successfully.")

    # -------------------------------
    # Main RAG Query
    # -------------------------------
    def query(self, question: str, k: int = 3) -> str:
        if not self._loaded:
            self._load_resources()

        query_vec = self._embedder.encode([question])[0].astype("float32")

        distances, indices = self._index.search(np.array([query_vec]), k)

        retrieved_text = "\n".join(
            self._chunks[idx]["chunk"]
            for idx in indices[0]
            if idx < len(self._chunks)
        )

        prompt = f"""
You are a Loan Product Assistant for Bank of Maharashtra.
Answer ONLY using the following retrieved context:

=== Retrieved Context ===
{retrieved_text}
-------------------------

Question: {question}

Answer in plain simple text.
No markdown, no bullet points.
"""

        response = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content
        return self._strip_markdown(answer)


# -------------------------------
# Standalone CLI tester
# -------------------------------
if __name__ == "__main__":
    rag = RAGAssistant()

    print("\nRAG system ready!")
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        try:
            print("\nANSWER:\n", rag.query(q))
        except Exception as e:
            print("Error:", e)
