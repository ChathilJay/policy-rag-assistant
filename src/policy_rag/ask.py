"""Ask a question and get a cited answer: retrieval (hybrid search + reranking), then Claude.

Needs ANTHROPIC_API_KEY in a .env file at the repo root (see .env.example).
Run:  uv run python -m policy_rag.ask "Can my employer make me do an unpaid trial shift?"
"""

import sys

import anthropic
from dotenv import load_dotenv

from policy_rag.answer import answer
from policy_rag.embeddings import FastEmbedder
from policy_rag.hybrid import hybrid_search
from policy_rag.index import load_chunks, open_store
from policy_rag.keyword import KeywordIndex
from policy_rag.rerank import CrossEncoderReranker, rerank


def main() -> None:
    load_dotenv()
    question = " ".join(sys.argv[1:]) or "How much notice does my employer have to give me?"

    embedder = FastEmbedder()
    shortlist = hybrid_search(
        question, embedder, open_store(embedder.dim), KeywordIndex(load_chunks()), k=20
    )
    hits = rerank(question, shortlist, CrossEncoderReranker(), k=5)

    try:
        result = answer(question, hits, anthropic.Anthropic())
    except anthropic.AuthenticationError:
        sys.exit("Invalid API key: check ANTHROPIC_API_KEY in your .env file.")
    except anthropic.RateLimitError:
        sys.exit("Rate limited by the API: wait a minute and try again.")
    except anthropic.APIConnectionError:
        sys.exit("Couldn't reach the Claude API: check your internet connection.")

    print(result.text, "\n")
    for n, source in enumerate(result.sources, start=1):
        print(f"[{n}] {source.title}: {source.url}")


if __name__ == "__main__":
    main()
