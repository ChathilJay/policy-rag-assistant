from qdrant_client import QdrantClient
from test_retrieval import FakeEmbedder

from policy_rag.hybrid import hybrid_search, reciprocal_rank_fusion
from policy_rag.index import build_index
from policy_rag.keyword import KeywordIndex, tokenize
from policy_rag.store import Hit, VectorStore

CHUNKS = [
    {"id": "fwis-0", "text": "Every new employee must receive the Fair Work Information Statement"},
    {"id": "notice-0", "text": "An employer must give written notice of termination"},
    {"id": "casual-0", "text": "Casual employees receive a casual loading instead of leave"},
]


def hit(cid: str) -> Hit:
    return Hit(score=0.0, chunk={"id": cid})


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the Fair Work Information Statement?") == [
        "fair",
        "work",
        "information",
        "statement",
    ]


def test_keyword_search_finds_exact_terms():
    hits = KeywordIndex(CHUNKS).search("information statement", k=3)
    assert hits[0].chunk["id"] == "fwis-0"


def test_keyword_search_skips_chunks_with_no_shared_words():
    hits = KeywordIndex(CHUNKS).search("superannuation", k=3)
    assert hits == []


def test_rrf_rewards_chunks_ranked_well_in_both_lists():
    dense = [hit("a"), hit("b"), hit("c")]
    keyword = [hit("b"), hit("c"), hit("a")]
    fused = reciprocal_rank_fusion([dense, keyword])
    assert [h.chunk["id"] for h in fused][0] == "b"  # 2nd + 1st beats 1st + 3rd


def test_rrf_keeps_chunks_found_by_only_one_method():
    fused = reciprocal_rank_fusion([[hit("a")], [hit("z")]])
    assert {h.chunk["id"] for h in fused} == {"a", "z"}


def test_hybrid_search_returns_k_fused_results():
    embedder = FakeEmbedder()
    store = VectorStore(QdrantClient(":memory:"), dim=embedder.dim)
    build_index(CHUNKS, embedder, store)
    hits = hybrid_search(
        "written notice of termination", embedder, store, KeywordIndex(CHUNKS), k=2
    )
    assert len(hits) == 2
    assert hits[0].chunk["id"] == "notice-0"
