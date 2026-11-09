# Dev log

One entry per working day: what was built, why, and what to revise. Newest first.

## Day 1 — 9 November 2026: setup, documents and ingestion

**Built**
- Project skeleton with `uv`, `pytest` and `ruff` (`pyproject.toml`, `.gitignore`, `.env.example`).
- `data/sources.csv`: 13 Fair Work Ombudsman documents (5 PDFs, 8 web pages).
- `scripts/download.py`: fetches them into `data/raw/`, skipping files already there.
- `src/policy_rag/ingest.py`: turns each document into text and splits it into chunks.
- Fix: the first chunk of every web page held about 250 words of site menu. The loader now
  keeps only the element with `id="main-content"`, which the page's "Skip to main content"
  link points to.

**Concepts to revise**
- **Why chunk at all?** Embedding models and LLM prompts have size limits, and a small chunk
  matches a question more precisely than a whole document does.
- **Chunk size trade-off.** Too small loses context; too large dilutes the match and wastes
  prompt space. 450 words (about 600 tokens) is a common middle ground.
- **Overlap.** Each chunk repeats the last 75 words of the previous one so a sentence split
  at a boundary still appears whole somewhere.
- **Tokens vs words.** Models count tokens; English averages about 0.75 words per token, which
  is why sizes are set in words here as a dependency-free estimate.
- **Metadata.** Every chunk keeps its title and URL so answers can cite their source.
- **Garbage in, garbage out.** Menu text repeated across pages would match almost any query.
  Always inspect your chunks before embedding them.

**Interview questions you should be able to answer**
1. How did you choose your chunk size and overlap, and how would you test a different one?
2. What happens to retrieval if boilerplate text ends up in your chunks?
3. Why are the raw documents gitignored instead of committed?

**Run it yourself**
```bash
uv run python scripts/download.py
uv run python -m policy_rag.ingest
uv run pytest
```
