# Dev log

One entry per working day: what was built, why, and what to revise. Newest first.

## Day 5 — 12 November 2026: answers with citations

**Built**
- `src/policy_rag/answer.py`: sends the top 5 reranked chunks and the question to Claude
  (`claude-opus-5-5`) and turns the reply into an answer with numbered sources.
- `src/policy_rag/ask.py`: the full pipeline in one command: hybrid search, reranking, answer.
- Tests with a fake API client, so they run without a key or network.

**Concepts to revise**
- **This is the "G" in RAG.** Retrieval finds the evidence; generation writes the answer from
  it. The system prompt tells Claude to answer only from the documents and to say when they
  don't cover the question, which is how you limit made-up answers (hallucination).
- **Native citations.** Each chunk is sent as a `document` block with
  `citations: {"enabled": true}`. The API splits the reply into text blocks, each listing the
  exact quote it relies on (`cited_text`) and which document (`document_index`). The quotes
  are extracted by the API, so they always exist in your documents, unlike asking the model
  to write "[1]" itself.
- **Prompt layout.** Long documents go first and the question last; models answer more
  accurately that way.
- **Effort.** `output_config.effort` trades thinking depth against cost and latency. Short Q&A
  over provided text doesn't need deep reasoning, so it is set to `medium`.
- **Refusals and fallbacks.** If the model declines on safety grounds the reply has
  `stop_reason: "refusal"`; the code handles that, and the `fallbacks` setting lets the API
  retry on another model automatically.
- **Secrets.** The API key lives in `.env`, which is gitignored and loaded with `python-dotenv`.
  It never goes in code or commits.

**Interview questions you should be able to answer**
1. How does your assistant avoid making things up, and how would you measure whether it does?
2. Why use the API's citations instead of asking the model to add [1] markers?
3. What happens when retrieval finds nothing relevant?

**Run it yourself**
1. Get an API key at platform.claude.com, copy `.env.example` to `.env`, and paste the key in.
2. Then:
```bash
uv sync
uv run python -m policy_rag.ask "Can my employer make me do an unpaid trial shift?"
```
Each question costs a fraction of a cent at this size.

## Day 4 — 11 November 2026: reranking

**Built**
- `src/policy_rag/rerank.py`: a `Reranker` interface and `CrossEncoderReranker`, which runs the
  `ms-marco-MiniLM-L-6-v2` cross-encoder locally through fastembed.
- Search now works in two stages: hybrid search finds 20 candidates, then the reranker
  re-scores them and keeps the best 5. `--no-rerank` skips the second stage for comparison.

**Concepts to revise**
- **Bi-encoder vs cross-encoder.** The embedding model (bi-encoder) encodes question and chunk
  separately and compares vectors, which is fast enough to search everything. A cross-encoder
  reads question and chunk together, so it notices whether the chunk actually answers the
  question, but it costs one model run per chunk.
- **Retrieve then rerank.** Use the fast method for recall (don't miss the right chunk among
  20), then the accurate method for precision (put it first). This two-stage pattern is
  standard in search and RAG.
- **Why order matters for RAG.** Only the top few chunks go into the LLM prompt. A correct chunk
  at rank 9 is as useless as a missing one.
- **Latency budget.** Reranking 20 short chunks on CPU takes well under a second; reranking all
  82 on every query would waste time for little gain.

**Interview questions you should be able to answer**
1. Why not use the cross-encoder for the whole search?
2. How would you decide how many candidates (20 here) to pass to the reranker?
3. How would you measure whether reranking actually helped?

**Run it yourself**
```bash
uv run python -m policy_rag.search --no-rerank "Can I be asked to work unpaid overtime?"
uv run python -m policy_rag.search "Can I be asked to work unpaid overtime?"
```
The first reranked search downloads the model (about 80 MB).

## Day 3 — 10 November 2026: hybrid search

**Built**
- `src/policy_rag/keyword.py`: BM25 keyword search over the same chunks.
- `src/policy_rag/hybrid.py`: runs meaning-based and keyword search, then merges the two
  ranked lists with Reciprocal Rank Fusion (RRF).
- `search.py` now takes `--mode hybrid|dense|keyword`, so you can compare all three.

**Concepts to revise**
- **Why hybrid?** Embeddings handle paraphrase ("fired" finds "dismissal") but can blur exact
  terms like a document name, a section number or a dollar amount. BM25 is the opposite.
  Combining them covers both failure modes, and most production RAG systems do this.
- **BM25.** Scores a chunk by the question words it contains, giving more weight to rare words
  (high IDF) and less to words repeated many times in one chunk (term saturation).
- **Stopwords.** Words like "the" and "how" appear everywhere and carry no signal, so they are
  dropped before scoring.
- **Reciprocal Rank Fusion.** Each list gives a chunk 1 / (60 + rank). Summing across lists
  rewards chunks that rank well in both. It uses ranks, not scores, because cosine scores
  (0 to 1) and BM25 scores (unbounded) are not comparable.
- **Candidates vs k.** Each method fetches 20 candidates, and fusion picks the best 5, so a chunk
  ranked 12th by one method can still win if the other ranks it highly.

**Interview questions you should be able to answer**
1. Give a question where keyword search beats embeddings, and one where it loses.
2. Why not just add the cosine score and the BM25 score together?
3. What does the 60 in RRF control, and what happens if it is very small?

**Run it yourself**
```bash
uv sync
uv run python -m policy_rag.search --mode dense "Fair Work Information Statement"
uv run python -m policy_rag.search --mode keyword "Fair Work Information Statement"
uv run python -m policy_rag.search "Fair Work Information Statement"
```

## Day 2 — 9 November 2026: embeddings and vector search

**Built**
- `src/policy_rag/embeddings.py`: a small `Embedder` interface and `FastEmbedder`, which runs
  the `BAAI/bge-small-en-v1.5` model locally through fastembed (no GPU or PyTorch needed).
- `src/policy_rag/store.py`: `VectorStore`, a wrapper around a Qdrant collection.
- `src/policy_rag/index.py`: embeds every chunk and saves the index to `data/index/`.
- `src/policy_rag/search.py`: embeds a question and returns the 5 closest chunks.
- Tests with a fake embedder, so they run in a second without downloading a model.

**Concepts to revise**
- **Embedding.** A model maps text to a list of numbers (here 384) so that texts with similar
  meaning land close together. "Fired" and "dismissed" end up near each other even though
  they share no letters, which keyword search cannot do.
- **Cosine similarity.** Measures the angle between two vectors: 1 means same direction (same
  meaning), 0 means unrelated. Length is ignored, so long and short chunks compare fairly.
- **Query vs passage embeddings.** BGE models were trained with different prefixes for
  questions and documents; `query_embed` and `passage_embed` add them for you.
- **Vector database.** Stores vectors plus metadata (title, URL) and finds nearest neighbours
  fast. Qdrant's embedded mode is just a folder on disk; the same code works against a Qdrant
  server later by changing one line.
- **Dependency injection.** Indexing and search take an `Embedder`, not a specific model, so
  tests pass a fake and you could swap in an API embedding model without other changes.

**Interview questions you should be able to answer**
1. Why use embeddings instead of keyword search, and when does keyword search still win?
2. Why must questions and documents be embedded by the same model?
3. Why does `build_index` recreate the collection instead of adding to it?

**Run it yourself**
```bash
uv sync
uv run python -m policy_rag.index
uv run python -m policy_rag.search "Can my employer make me do an unpaid trial shift?"
```
The first run downloads the model (about 130 MB).

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
