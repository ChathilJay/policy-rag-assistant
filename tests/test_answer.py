from types import SimpleNamespace as NS

from policy_rag.answer import NO_RESULTS, answer, build_content, parse_response
from policy_rag.store import Hit

HITS = [
    Hit(0.9, {"title": "Notice of termination", "url": "https://x/notice", "text": "Notice A."}),
    Hit(0.8, {"title": "Casual employees", "url": "https://x/casual", "text": "Casual B."}),
]


def text(t, *docs):
    citations = [NS(document_index=d, cited_text=f" quote from {d} ") for d in docs]
    return NS(type="text", text=t, citations=citations or None)


def response(*blocks, stop_reason="end_turn"):
    return NS(content=list(blocks), stop_reason=stop_reason)


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def test_build_content_sends_each_chunk_as_a_citable_document():
    content = build_content("How much notice?", HITS)
    assert [c["type"] for c in content] == ["document", "document", "text"]
    assert content[0]["citations"] == {"enabled": True}
    assert content[0]["title"] == "Notice of termination"
    assert content[-1]["text"] == "How much notice?"


def test_parse_numbers_sources_in_order_of_first_citation():
    resp = response(text("You get notice", 1), text(" and pay.", 0), text(" Also notice.", 1))
    result = parse_response(resp, HITS)
    assert result.text == "You get notice[1] and pay.[2] Also notice.[1]"
    assert [s.title for s in result.sources] == ["Casual employees", "Notice of termination"]
    assert result.sources[0].quotes == ["quote from 1", "quote from 1"]


def test_parse_ignores_thinking_blocks_and_uncited_text():
    resp = response(NS(type="thinking", thinking=""), text("Plain answer."))
    result = parse_response(resp, HITS)
    assert result.text == "Plain answer."
    assert result.sources == []


def test_parse_handles_refusal():
    result = parse_response(response(stop_reason="refusal"), HITS)
    assert result.refused and result.sources == []


def test_answer_without_hits_skips_the_api():
    client = FakeClient(reply=None)
    assert answer("anything", [], client).text == NO_RESULTS
    assert client.calls == []


def test_answer_sends_documents_and_returns_cited_text():
    client = FakeClient(response(text("Two weeks.", 0)))
    result = answer("How much notice?", HITS, client)
    assert result.text == "Two weeks.[1]"
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert len(call["messages"][0]["content"]) == 3
