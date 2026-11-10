"""Keyword search with BM25, the ranking formula behind classic search engines.

Embeddings capture meaning but can miss exact terms such as "Fair Work Information
Statement" or a specific dollar figure. BM25 scores chunks by the words they share with
the question, weighting rare words more than common ones.
"""

import re

from rank_bm25 import BM25Okapi

from policy_rag.store import Hit

STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "can", "do", "does", "for", "how", "i", "if", "in",
    "is", "it", "me", "my", "of", "on", "or", "the", "to", "what", "when", "who", "with",
}  # fmt: skip


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class KeywordIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, question: str, k: int = 5) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [
            Hit(score=float(scores[i]), chunk=self.chunks[i]) for i in ranked[:k] if scores[i] > 0
        ]
