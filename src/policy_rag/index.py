"""Embed every chunk and store it in a local Qdrant index.

Run after ingest:  uv run python -m policy_rag.index
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient

from policy_rag.embeddings import Embedder, FastEmbedder
from policy_rag.store import VectorStore

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/index")  # Qdrant's embedded mode: a folder on disk, no server


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_index(chunks: list[dict], embedder: Embedder, store: VectorStore) -> int:
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    store.recreate()
    store.add(chunks, vectors)
    return len(chunks)


def open_store(dim: int) -> VectorStore:
    return VectorStore(QdrantClient(path=str(INDEX_PATH)), dim=dim)


if __name__ == "__main__":
    embedder = FastEmbedder()
    n = build_index(load_chunks(), embedder, open_store(embedder.dim))
    print(f"Indexed {n} chunks into {INDEX_PATH}")
