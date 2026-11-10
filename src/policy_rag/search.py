"""Find the chunks most relevant to a question.

Run after indexing:
    uv run python -m policy_rag.search "How much notice do I get?"
    uv run python -m policy_rag.search --mode keyword "Fair Work Information Statement"

Modes: hybrid (default), dense (meaning only), keyword (BM25 only).
"""

import argparse

from policy_rag.embeddings import Embedder, FastEmbedder
from policy_rag.store import Hit, VectorStore


def search(question: str, embedder: Embedder, store: VectorStore, k: int = 5) -> list[Hit]:
    return store.search(embedder.embed_query(question), k=k)


def main() -> None:
    from policy_rag.hybrid import hybrid_search
    from policy_rag.index import load_chunks, open_store
    from policy_rag.keyword import KeywordIndex

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mode", choices=["hybrid", "dense", "keyword"], default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    question = " ".join(args.question)

    if args.mode == "keyword":
        hits = KeywordIndex(load_chunks()).search(question, k=args.k)
    else:
        embedder = FastEmbedder()
        store = open_store(embedder.dim)
        if args.mode == "dense":
            hits = search(question, embedder, store, k=args.k)
        else:
            hits = hybrid_search(question, embedder, store, KeywordIndex(load_chunks()), k=args.k)

    for hit in hits:
        print(f"{hit.score:.4f}  {hit.chunk['title']}  ({hit.chunk['id']})")
        print(f"        {hit.chunk['text'][:160]}...\n")


if __name__ == "__main__":
    main()
