import json

import pytest

from policy_rag.ingest import chunk_text, ingest, load_html


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_chunks_never_exceed_the_word_limit():
    chunks = chunk_text(words(2000), chunk_words=450, overlap=75)
    assert all(len(c.split()) <= 450 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(words(1000), chunk_words=100, overlap=20)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.split()[-20:] == nxt.split()[:20]


def test_no_words_are_lost():
    text = words(1234)
    chunks = chunk_text(text, chunk_words=100, overlap=20)
    rebuilt = chunks[0].split()
    for c in chunks[1:]:
        rebuilt += c.split()[20:]
    assert rebuilt == text.split()


def test_short_text_is_one_chunk():
    assert chunk_text("Casual employees can request conversion.") == [
        "Casual employees can request conversion."
    ]


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_text(words(10), chunk_words=5, overlap=5)


def test_html_loader_drops_navigation_and_scripts(tmp_path):
    page = tmp_path / "page.html"
    page.write_text(
        "<html><body><nav>Home Menu</nav><script>track()</script>"
        "<main><p>Notice periods depend on length of service.</p></main>"
        "<footer>Copyright</footer></body></html>"
    )
    text = load_html(page)
    assert "Notice periods depend on length of service." in text
    for noise in ("Home Menu", "track()", "Copyright"):
        assert noise not in text


def test_html_loader_prefers_main_content_id_over_site_chrome(tmp_path):
    # Shape of a fairwork.gov.au page: the menu sits outside #main-content and there is no <nav>.
    page = tmp_path / "page.html"
    page.write_text(
        '<html><body><a href="#main-content">Skip to main content</a>'
        '<div class="dialog-off-canvas-main-canvas">'
        '<div class="block-websitetranslation">Translate this website Bengali Bosnian</div>'
        '<div id="main-content"><h1>Dismissal</h1><p>An employer must give notice.</p></div>'
        "</div></body></html>"
    )
    text = load_html(page)
    assert "An employer must give notice." in text
    assert "Translate this website" not in text
    assert "Skip to main content" not in text


def test_ingest_writes_chunks_with_source_metadata(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dismissal.html").write_text(f"<main><p>{words(600)}</p></main>")
    (raw / "unlisted.html").write_text("<main><p>not in sources</p></main>")
    sources = tmp_path / "sources.csv"
    sources.write_text("id,title,url\ndismissal,Dismissal,https://example.org/dismissal\n")
    out = tmp_path / "chunks.jsonl"

    count = ingest(raw_dir=raw, sources=sources, out_path=out)

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert count == len(records) == 2
    assert {r["source_id"] for r in records} == {"dismissal"}
    assert records[0]["title"] == "Dismissal"
    assert records[1]["id"] == "dismissal-1"
