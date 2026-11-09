"""A thin wrapper around a Qdrant collection holding chunk vectors and their metadata."""

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION = "fair_work"


@dataclass
class Hit:
    score: float
    chunk: dict


class VectorStore:
    def __init__(self, client: QdrantClient, dim: int, collection: str = COLLECTION):
        self.client = client
        self.dim = dim
        self.collection = collection

    def recreate(self) -> None:
        """Start from an empty collection, so re-indexing never leaves stale chunks behind."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        # Cosine similarity compares the direction of vectors, not their length.
        self.client.create_collection(
            self.collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
        )

    def add(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(id=i, vector=vector, payload=chunk)
            for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, vector: list[float], k: int = 5) -> list[Hit]:
        result = self.client.query_points(self.collection, query=vector, limit=k)
        return [Hit(score=p.score, chunk=p.payload) for p in result.points]
