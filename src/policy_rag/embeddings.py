"""Embedding models: turn text into vectors whose distance reflects meaning.

Code depends on the small `Embedder` protocol rather than a specific library, so the real
model can be swapped (or faked in tests) without touching the indexing or search code.
"""

from typing import Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, ~130 MB, runs on CPU


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Local embedding model via fastembed (ONNX runtime, no PyTorch or GPU needed).

    The model is downloaded from Hugging Face on first use and cached.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding  # imported here so tests don't load the model

        self.model = TextEmbedding(model_name)
        self.dim = len(self.embed_query("probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # passage_embed and query_embed add the prefixes BGE models were trained with.
        return [v.tolist() for v in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()
