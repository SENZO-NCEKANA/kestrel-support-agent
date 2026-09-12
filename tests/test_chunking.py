"""
Chunking tests.

The critical assertion is table integrity. Everything downstream depends on
it: if the limits table splits between the verification rows and the tier
rows, demo scenario 1 cannot work regardless of prompt quality.
"""
import re
from pathlib import Path

import pytest

from kestrel.chunking import MAX_CHARS, chunk_directory, chunk_document, parse_frontmatter

KB = Path(__file__).resolve().parents[1] / "kb"

TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


@pytest.fixture(scope="module")
def chunks():
    return chunk_directory(KB)


def table_blocks(text: str) -> list[list[str]]:
    """Extract contiguous runs of table rows."""
    blocks, current = [], []
    for line in text.splitlines():
        if TABLE_ROW.match(line):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def test_corpus_produces_chunks(chunks):
    assert len(chunks) > 20
    assert all(c.text.strip() for c in chunks)


def test_every_chunk_has_provenance(chunks):
    for c in chunks:
        assert c.doc_id.startswith("KB-"), f"missing doc_id: {c.doc_id}"
        assert c.doc_title
        assert c.metadata.get("version")


def test_no_orphaned_table_fragments(chunks):
    """A table run must carry its header, never start mid-body.

    This is the test that catches the highest-impact chunking bug: a run of
    data rows with no header row, which reads to the model as a list of
    unlabelled numbers.
    """
    for c in chunks:
        for block in table_blocks(c.text):
            if len(block) < 2:
                continue
            assert TABLE_SEP.match(block[1]), (
                f"table fragment without header separator in {c.doc_id} "
                f"chunk {c.chunk_index}:\n" + "\n".join(block[:3])
            )


def test_limits_tables_survive_intact():
    """Both limit tables must each land wholly inside a single chunk.

    Scenario 1 requires the model to compare a verification-level limit
    against a tier ceiling. If either table is split, the comparison is
    impossible.
    """
    chunks = chunk_document((KB / "03-transaction-limits.md").read_text())

    def chunk_containing(*needles):
        for c in chunks:
            if all(n in c.text for n in needles):
                return c
        return None

    # Verification-level table: Level 1 ATM through international transfer row
    verif = chunk_containing("ATM withdrawal", "Level 1", "International transfer")
    assert verif is not None, "verification-level limits table was split"
    assert "R2 000" in verif.text and "R10 000" in verif.text

    # Tier ceiling table
    tier = chunk_containing("Blue", "Plus", "Private", "EFT / PayShap out")
    assert tier is not None, "tier ceiling table was split"


def test_worked_example_stays_with_its_table():
    """The worked example explains the precedence rule and must be retrievable."""
    chunks = chunk_document((KB / "03-transaction-limits.md").read_text())
    example = [c for c in chunks if "Worked example" in c.text]
    assert example, "worked example missing"
    assert "verification level is the binding constraint" in example[0].text


def test_fraud_exception_not_separated_from_fee_table():
    """The fraud override must sit with the card fee table it overrides."""
    chunks = chunk_document((KB / "01-fee-schedule.md").read_text())
    fraud = [c for c in chunks if "free on all tiers" in c.text]
    assert fraud, "fraud override line missing"
    assert "Replacement card" in fraud[0].text, (
        "fraud override was separated from the card fee table it modifies"
    )


def test_heading_path_present(chunks):
    with_path = [c for c in chunks if c.heading_path]
    assert len(with_path) / len(chunks) > 0.9


def test_embed_text_includes_context(chunks):
    c = chunks[5]
    assert c.doc_title in c.embed_text
    assert c.text in c.embed_text


def test_no_chunk_exceeds_hard_max(chunks):
    for c in chunks:
        assert len(c.text) <= MAX_CHARS * 1.5, (
            f"{c.doc_id} chunk {c.chunk_index} is {len(c.text)} chars"
        )


def test_chunk_ids_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_content_hash_is_stable():
    a = chunk_document((KB / "01-fee-schedule.md").read_text())
    b = chunk_document((KB / "01-fee-schedule.md").read_text())
    assert [c.content_hash for c in a] == [c.content_hash for c in b]


def test_frontmatter_parsing():
    meta, body = parse_frontmatter(
        "---\ndoc_id: KB-TEST-001\ntitle: Test Doc\nversion: 1.0\n---\n\n# Heading\n\nBody.\n"
    )
    assert meta["doc_id"] == "KB-TEST-001"
    assert meta["title"] == "Test Doc"
    assert body.lstrip().startswith("# Heading")


def test_oversized_table_repeats_header():
    rows = "\n".join(f"| Item {i} | Value {i} | {'x' * 60} |" for i in range(120))
    doc = f"---\ndoc_id: KB-BIG-001\ntitle: Big\nversion: 1\n---\n\n# Big\n\n| A | B | C |\n|---|---|---|\n{rows}\n"
    chunks = chunk_document(doc)
    table_chunks = [c for c in chunks if "| A | B | C |" in c.text]
    assert len(table_chunks) > 1, "oversized table should have been split"
    for c in table_chunks:
        assert c.text.count("| A | B | C |") == 1
