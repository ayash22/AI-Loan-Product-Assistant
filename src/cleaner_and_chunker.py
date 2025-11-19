# ----------------------------------------------------------
# cleaner_and_chunker.py — Build the Knowledge Base (clean + chunk text)
# ----------------------------------------------------------

import json
import re
from pathlib import Path

# ------------------------------
# Folders / Files
# ------------------------------
PROCESSED_DIR = Path("../data/processed")
KB_DIR = Path("../data/knowledge_base")
KB_DIR.mkdir(parents=True, exist_ok=True)

KB_JSONL = KB_DIR / "chunks.json"   # For RAG pipeline
KB_TXT = KB_DIR / "kb.txt"          # Full readable KB

# ------------------------------
# Chunking parameters
# ------------------------------
CHUNK_SIZE = 450   # characters
CHUNK_OVERLAP = 100

# ------------------------------
# Helper: Load all cleaned text files
# ------------------------------
def load_scraped_text():
    files = [f for f in PROCESSED_DIR.glob("*.txt") if f.name not in ("kb.txt", "kb.jsonl")]
    documents = []

    for f in files:
        txt = f.read_text(encoding="utf-8").strip()
        if txt:
            documents.append((f.name, txt))
    return documents

# ------------------------------
# Helper: Clean text (remove leftover noise)
# ------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    text = text.replace("\t", " ")
    return text.strip()

# ------------------------------
# Chunk text into overlapping segments
# ------------------------------
def create_chunks(text: str, doc_name: str) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        chunks.append({
            "doc": doc_name,
            "chunk": chunk_text,
            "start": start,
            "end": end
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def build_kb():
    documents = load_scraped_text()
    all_chunks = []
    kb_text = ""

    print(f"Found {len(documents)} scraped text files.")

    for doc_name, txt in documents:
        clean = clean_text(txt)
        # Add to master readable KB file
        kb_text += f"\n\n===== {doc_name} =====\n{clean}\n"

        # Chunk text for embedding + RAG
        chunks = create_chunks(clean, doc_name)
        all_chunks.extend(chunks)

    # ---- Save full readable KB
    KB_TXT.write_text(kb_text, encoding="utf-8")

    # ---- Save chunks for RAG
    with open(KB_JSONL, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"✔ Knowledge base built successfully.")
    print(f"✔ Full KB saved to: {KB_TXT}")
    print(f"✔ Chunks saved to: {KB_JSONL}")
    print(f"Total chunks: {len(all_chunks)}")

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    build_kb()
