from policy_rag.rerank import rerank
from policy_rag.store import Hit


class FakeReranker:
    """Scores a chunk by how many question words it contains, standing in for a cross-encoder."""

    def __init__(self):
        self.calls = 0

    def score(self, question, texts):
        self.calls += 1
        words = set(question.lower().split())
        return [float(len(words & set(t.lower().split()))) for t in texts]


def hits(*texts):
    return [Hit(score=0.0, chunk={"id": str(i), "text": t}) for i, t in enumerate(texts)]


def test_rerank_reorders_by_new_score():
    candidates = hits("casual loading rate", "notice of termination period", "annual leave")
    top = rerank("termination notice period", candidates, FakeReranker(), k=3)
    assert top[0].chunk["text"] == "notice of termination period"


def test_rerank_keeps_only_k_results():
    candidates = hits("a", "b", "c", "d")
    assert len(rerank("a", candidates, FakeReranker(), k=2)) == 2


def test_rerank_scores_the_whole_shortlist_in_one_call():
    reranker = FakeReranker()
    rerank("q", hits("a", "b", "c"), reranker)
    assert reranker.calls == 1


def test_rerank_of_nothing_is_nothing():
    assert rerank("q", [], FakeReranker()) == []
