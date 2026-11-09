"""Download the documents listed in data/sources.csv into data/raw/.

Run from the repo root:  uv run python scripts/download.py
Files that already exist are skipped, so it is safe to re-run.
"""

import csv
from pathlib import Path

import httpx

SOURCES = Path("data/sources.csv")
RAW_DIR = Path("data/raw")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with SOURCES.open(newline="") as f:
        rows = list(csv.DictReader(f))

    with httpx.Client(follow_redirects=True, timeout=30) as client:
        for row in rows:
            suffix = ".pdf" if ".pdf" in row["url"].lower() else ".html"
            path = RAW_DIR / f"{row['id']}{suffix}"
            if path.exists():
                print(f"skip  {path.name}")
                continue
            response = client.get(row["url"])
            response.raise_for_status()
            path.write_bytes(response.content)
            print(f"saved {path.name} ({len(response.content) // 1024} KB)")


if __name__ == "__main__":
    main()
