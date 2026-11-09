# Policy RAG Assistant

A question-answering assistant for Australian workplace rights. It answers questions such as
"How much notice does my employer have to give me?" or "Can I be asked to do an unpaid trial?"
using Fair Work Ombudsman fact sheets, and cites the document each answer comes from.

Built for students, graduates and casual workers who want a quick, sourced answer before
reading the full guidance.

> Status: in progress. Week 1 covers project setup, the document list and ingestion.

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
