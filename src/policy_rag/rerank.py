"""Reranking: re-score the shortlist from hybrid search with a more accurate, slower model.

An embedding model encodes the question and each chunk separately, then compares vectors.
A cross-encoder reads the question and one chunk together, so it can judge whether that
chunk actually answers the question. It is too slow to run on every chunk, so it only
re-scores the ~20 candidates hybrid search already found.
"""

from typing import Protocol

from policy_rag.store import Hit

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # ~80 MB, trained on Bing question/passage pairs


class Reranker(Protocol):
    def score(self, question: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name)

    def score(self, question: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(question, texts)]


def rerank(question: str, hits: list[Hit], reranker: Reranker, k: int = 5) -> list[Hit]:
    if not hits:
        return []
    scores = reranker.score(question, [h.chunk["text"] for h in hits])
    rescored = [Hit(score=s, chunk=h.chunk) for s, h in zip(scores, hits, strict=True)]
    return sorted(rescored, key=lambda h: h.score, reverse=True)[:k]
