# Policy RAG Assistant

A question-answering assistant for Australian workplace rights. It answers questions such as
"How much notice does my employer have to give me?" or "Can I be asked to do an unpaid trial?"
using Fair Work Ombudsman fact sheets, and cites the document each answer comes from.

Built for students, graduates and casual workers who want a quick, sourced answer before
reading the full guidance.

> Status: in progress. See [docs/devlog.md](docs/devlog.md) for what was built each day.

## Setup

```bash
uv sync
uv run pytest
```

## Documents

The assistant answers from 13 public Fair Work Ombudsman documents listed in
[`data/sources.csv`](data/sources.csv): fact sheets on the National Employment Standards,
casual work, notice and redundancy, dismissal, modern awards and international students.
They are downloaded rather than committed:

```bash
uv run python scripts/download.py
```

## Ingestion

```bash
uv run python -m policy_rag.ingest
```

Each document is converted to plain text (PDFs with `pypdf`, web pages with BeautifulSoup,
with navigation and scripts stripped) and split into chunks of about 450 words, roughly 600
tokens, with a 75-word overlap so a sentence cut at a boundary still appears whole in one chunk.
Chunks are written to `data/processed/chunks.jsonl` with their source title and URL, so later
answers can cite where they came from.

## Search

```bash
uv run python -m policy_rag.index
uv run python -m policy_rag.search "How much notice does my employer have to give me?"
```

Search is hybrid. Chunks are embedded locally with `BAAI/bge-small-en-v1.5` (via fastembed)
and stored in an embedded Qdrant index in `data/index/`, and a BM25 keyword index is built
over the same chunks. Each returns 20 candidates, which are merged with Reciprocal Rank Fusion
into a shortlist of 20. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) then re-reads each
candidate alongside the question and keeps the best five. Use `--mode dense` or
`--mode keyword` to see either method alone, or `--no-rerank` to skip the last step.

## Ask a question

```bash
cp .env.example .env   # then add your ANTHROPIC_API_KEY
uv run python -m policy_rag.ask "Can my employer make me do an unpaid trial shift?"
```

The five best chunks are sent to Claude as separate documents with citations enabled, so every
statement in the answer links back to the exact passage it came from. If the documents don't
cover the question, the assistant says so and points to the Fair Work Infoline.
