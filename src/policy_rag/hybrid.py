"""Hybrid search: combine meaning-based and keyword-based results with Reciprocal Rank Fusion."""

from policy_rag.embeddings import Embedder
from policy_rag.keyword import KeywordIndex
from policy_rag.store import Hit, VectorStore

RRF_K = 60  # the constant from the original RRF paper; dampens the gap between top ranks


def reciprocal_rank_fusion(result_lists: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    """Merge ranked lists by rank, not score.

    Dense scores (cosine, 0 to 1) and BM25 scores (unbounded) are on different scales, so
    adding them directly would let one dominate. RRF gives each chunk 1 / (k + rank) per list
    it appears in, which rewards chunks that rank well in both.
    """
    fused: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk["id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            chunks[cid] = hit.chunk
    order = sorted(fused, key=fused.get, reverse=True)
    return [Hit(score=fused[cid], chunk=chunks[cid]) for cid in order]


def hybrid_search(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    keyword_index: KeywordIndex,
    k: int = 5,
    candidates: int = 20,
) -> list[Hit]:
    dense = store.search(embedder.embed_query(question), k=candidates)
    keyword = keyword_index.search(question, k=candidates)
    return reciprocal_rank_fusion([dense, keyword])[:k]
