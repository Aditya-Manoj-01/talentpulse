from rag_service.pipeline import RagConfig, RagPipeline, chunk_document


def test_markdown_chunks_carry_headings():
    cfg = RagConfig(chunker="markdown", chunk_size=200, chunk_overlap=20)
    chunks = chunk_document("# Title\n\n## Limits\nA. B. C.\n\n## Pay\nD.", "x.md", cfg)
    assert [c.section for c in chunks] == ["Limits", "Pay"]
    assert chunks[0].text.startswith("Title — Limits:")


def test_retrieves_correct_source_offline():
    pipe = RagPipeline(RagConfig(embedder="tfidf-lsa"))
    hits = pipe.retrieve("What is the weekly overtime cap?")
    assert hits and hits[0]["source"] == "overtime_policy.md"
