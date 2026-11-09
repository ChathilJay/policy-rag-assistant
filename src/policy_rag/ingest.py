"""Turn downloaded documents into overlapping text chunks ready for embedding.

Run from the repo root after downloading:  uv run python -m policy_rag.ingest
Output: data/processed/chunks.jsonl, one JSON object per chunk.
"""

import csv
import json
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

RAW_DIR = Path("data/raw")
SOURCES = Path("data/sources.csv")
OUT_PATH = Path("data/processed/chunks.jsonl")

# Chunk sizes are in words. English text averages about 0.75 words per token,
# so 450 words is roughly 600 tokens and the 75-word overlap roughly 100 tokens.
CHUNK_WORDS = 450
OVERLAP_WORDS = 75


def load_pdf(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    # Site navigation, scripts and footers would pollute every chunk with the same text.
    for tag in soup(["script", "style", "nav", "header", "footer", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    return main.get_text(separator="\n")


def load_document(path: Path) -> str:
    if path.suffix == ".pdf":
        return load_pdf(path)
    if path.suffix in {".html", ".htm"}:
        return load_html(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def chunk_text(
    text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS
) -> list[str]:
    """Split text into chunks of at most `chunk_words` words.

    Each chunk repeats the last `overlap` words of the previous one, so a sentence
    cut at a boundary still appears whole in at least one chunk.
    """
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    words = text.split()
    chunks = []
    step = chunk_words - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


def ingest(raw_dir: Path = RAW_DIR, sources: Path = SOURCES, out_path: Path = OUT_PATH) -> int:
    with sources.open(newline="") as f:
        meta = {row["id"]: row for row in csv.DictReader(f)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in sorted(raw_dir.iterdir()):
            if path.stem not in meta:
                continue
            source = meta[path.stem]
            for i, chunk in enumerate(chunk_text(load_document(path))):
                record = {
                    "id": f"{path.stem}-{i}",
                    "source_id": path.stem,
                    "title": source["title"],
                    "url": source["url"],
                    "chunk_index": i,
                    "text": chunk,
                }
                out.write(json.dumps(record) + "\n")
                count += 1
    return count


if __name__ == "__main__":
    n = ingest()
    print(f"Wrote {n} chunks to {OUT_PATH}")
