"""Generate a grounded answer with citations from retrieved chunks, using the Claude API.

Each retrieved chunk is sent as a separate document with citations enabled, so Claude's
answer comes back split into text blocks that point at the exact passage they rely on.
The API extracts the quoted text itself, so a citation can never point at text that
isn't in the documents.
"""

from dataclasses import dataclass, field

from policy_rag.store import Hit

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You answer questions about Australian workplace rights for students, \
graduates and casual workers.

Answer only from the Fair Work Ombudsman documents provided with the question, and cite \
them. If the documents do not contain the answer, say so plainly and suggest calling the \
Fair Work Infoline on 13 13 94; do not fill the gap from general knowledge.

Keep answers short and practical: lead with the direct answer, then the key conditions or \
exceptions. This is general information, not legal advice."""

NO_RESULTS = (
    "I couldn't find anything about that in the Fair Work documents I have. "
    "The Fair Work Infoline (13 13 94) can help with specific situations."
)


@dataclass
class Source:
    title: str
    url: str
    quotes: list[str] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    sources: list[Source]
    refused: bool = False


def build_content(question: str, hits: list[Hit]) -> list[dict]:
    """One document block per chunk, then the question. Documents first, question last."""
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": hit.chunk["text"]},
            "title": hit.chunk["title"],
            "context": f"Source: {hit.chunk['url']}",
            "citations": {"enabled": True},
        }
        for hit in hits
    ]
    return [*documents, {"type": "text", "text": question}]


def parse_response(response, hits: list[Hit]) -> Answer:
    """Join the text blocks, adding [n] after each cited span, and collect the sources."""
    if response.stop_reason == "refusal":
        return Answer(text="Sorry, I can't help with that question.", sources=[], refused=True)

    sources: list[Source] = []
    number_of: dict[int, int] = {}  # document index -> citation number shown to the user
    parts: list[str] = []
    for block in response.content:
        if block.type != "text":
            continue  # thinking blocks and fallback markers are not part of the answer
        parts.append(block.text)
        markers = []
        for citation in getattr(block, "citations", None) or []:
            doc = citation.document_index
            if doc not in number_of:
                chunk = hits[doc].chunk
                sources.append(Source(title=chunk["title"], url=chunk["url"]))
                number_of[doc] = len(sources)
            sources[number_of[doc] - 1].quotes.append(citation.cited_text.strip())
            marker = f"[{number_of[doc]}]"
            if marker not in markers:
                markers.append(marker)
        parts.append("".join(markers))
    return Answer(text="".join(parts).strip(), sources=sources)


def answer(question: str, hits: list[Hit], client) -> Answer:
    if not hits:
        return Answer(text=NO_RESULTS, sources=[])
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        # Effort trades answer depth against cost and latency; medium suits short Q&A.
        output_config={"effort": "medium"},
        # If the model declines on safety grounds, the API retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": build_content(question, hits)}],
    )
    return parse_response(response, hits)
