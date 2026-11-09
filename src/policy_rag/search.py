"""Find the chunks closest in meaning to a question.

Run after indexing:  uv run python -m policy_rag.search "How much notice do I get?"
"""

import sys

from policy_rag.embeddings import Embedder, FastEmbedder
from policy_rag.store import Hit, VectorStore


def search(question: str, embedder: Embedder, store: VectorStore, k: int = 5) -> list[Hit]:
    return store.search(embedder.embed_query(question), k=k)


if __name__ == "__main__":
    from policy_rag.index import open_store

    question = " ".join(sys.argv[1:]) or "How much notice does my employer have to give me?"
    embedder = FastEmbedder()
    for hit in search(question, embedder, open_store(embedder.dim)):
        print(f"{hit.score:.3f}  {hit.chunk['title']}  ({hit.chunk['id']})")
        print(f"       {hit.chunk['text'][:160]}...\n")
