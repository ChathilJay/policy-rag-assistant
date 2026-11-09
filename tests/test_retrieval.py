import hashlib
import math

from qdrant_client import QdrantClient

from policy_rag.index import build_index
from policy_rag.search import search
from policy_rag.store import VectorStore


class FakeEmbedder:
    """Hashes each word into one of 64 slots: texts sharing words get similar vectors.

    Enough to test indexing and search logic without downloading a real model.
    """

    dim = 64

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in text.lower().split():
            slot = int(hashlib.md5(word.strip(".,?").encode()).hexdigest(), 16) % self.dim
            v[slot] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


CHUNKS = [
    {"id": "notice-0", "title": "Notice of termination", "text": "notice of termination period"},
    {"id": "casual-0", "title": "Casual employees", "text": "casual loading conversion rights"},
    {"id": "leave-0", "title": "Long service leave", "text": "long service leave entitlement"},
]


def make_store(embedder):
    return VectorStore(QdrantClient(":memory:"), dim=embedder.dim)


def test_build_index_stores_every_chunk():
    embedder = FakeEmbedder()
    store = make_store(embedder)
    assert build_index(CHUNKS, embedder, store) == 3
    assert store.client.count(store.collection).count == 3


def test_search_returns_the_most_similar_chunk_first():
    embedder = FakeEmbedder()
    store = make_store(embedder)
    build_index(CHUNKS, embedder, store)
    hits = search("what notice period applies to termination?", embedder, store, k=2)
    assert hits[0].chunk["id"] == "notice-0"
    assert hits[0].score > hits[1].score


def test_search_results_keep_source_metadata():
    embedder = FakeEmbedder()
    store = make_store(embedder)
    build_index(CHUNKS, embedder, store)
    hit = search("casual conversion", embedder, store, k=1)[0]
    assert hit.chunk["title"] == "Casual employees"


def test_rebuilding_replaces_the_old_index():
    embedder = FakeEmbedder()
    store = make_store(embedder)
    build_index(CHUNKS, embedder, store)
    build_index(CHUNKS[:1], embedder, store)
    assert store.client.count(store.collection).count == 1
