"""
test_chunker.py
---------------
Tests for services/chunker.py

What we test:
  - split_text_into_chunks : correct window size, overlap, edge cases
  - detect_repeated_headers: deduplication of short repeated headers
  - chunk_units            : full pipeline from units → chunks
"""

import pytest
from services.chunker import split_text_into_chunks, detect_repeated_headers, chunk_units


# ══════════════════════════════════════════════════════════════════════════════
#  split_text_into_chunks
# ══════════════════════════════════════════════════════════════════════════════

class TestSplitTextIntoChunks:

    def test_short_text_produces_single_chunk(self):
        """Text shorter than max_tokens should return exactly one chunk."""
        text = "Hello world. This is a short sentence."
        result = split_text_into_chunks(text, max_tokens=500, overlap=100)
        assert len(result) == 1
        assert "Hello world" in result[0]

    def test_long_text_produces_multiple_chunks(self):
        """Text longer than max_tokens must be split into multiple chunks."""
        # Generate a text that is definitely > 50 tokens
        text = " ".join(["word"] * 200)
        result = split_text_into_chunks(text, max_tokens=50, overlap=10)
        assert len(result) > 1

    def test_overlap_shares_tokens_between_chunks(self):
        """With overlap=10, the end of chunk N should appear at the start of chunk N+1."""
        words = [f"word{i}" for i in range(100)]
        text  = " ".join(words)
        chunks = split_text_into_chunks(text, max_tokens=20, overlap=10)

        # The last token of chunk 0 should appear somewhere in chunk 1
        last_token_of_first = chunks[0].split()[-1]
        assert last_token_of_first in chunks[1]

    def test_empty_text_returns_empty_list(self):
        result = split_text_into_chunks("", max_tokens=500, overlap=100)
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        result = split_text_into_chunks("   \n\t  ", max_tokens=500, overlap=100)
        assert result == []

    def test_chunks_are_strings(self):
        result = split_text_into_chunks("Some text here.", max_tokens=500, overlap=100)
        for chunk in result:
            assert isinstance(chunk, str)

    def test_no_chunk_exceeds_max_tokens(self):
        """Every chunk should be at or below the max_tokens limit."""
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        text  = " ".join(["hello"] * 1000)
        chunks = split_text_into_chunks(text, max_tokens=100, overlap=20)
        for chunk in chunks:
            token_count = len(tokenizer.encode(chunk))
            assert token_count <= 100, f"Chunk exceeded max_tokens: {token_count}"


# ══════════════════════════════════════════════════════════════════════════════
#  detect_repeated_headers
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectRepeatedHeaders:

    def _make_chunk(self, text, chunk_id="c1"):
        return {"chunk_id": chunk_id, "text": text, "document_id": "doc-1",
                "unit_id": "u1", "unit_number": 1, "unit_type": "pdf",
                "chunk_index": 0, "metadata": {}}

    def test_long_chunks_are_never_filtered(self):
        """Chunks with > 12 words should always pass through regardless of repetition."""
        long_text = " ".join(["word"] * 20)
        chunks = [self._make_chunk(long_text, f"c{i}") for i in range(10)]
        result = detect_repeated_headers(chunks)
        assert len(result) == 10

    def test_short_header_repeated_many_times_is_capped(self):
        """A short header appearing > 3 times should only keep the first 3 occurrences."""
        header = "Chapter Introduction"  # 2 words — qualifies as a short header
        chunks = [self._make_chunk(header, f"c{i}") for i in range(10)]
        result = detect_repeated_headers(chunks)
        # Should keep at most 3 occurrences
        kept_texts = [c["text"] for c in result]
        assert kept_texts.count(header) <= 3

    def test_unique_short_chunks_pass_through(self):
        """Short chunks that appear only once should not be filtered."""
        chunks = [
            self._make_chunk("Intro", "c1"),
            self._make_chunk("Summary", "c2"),
            self._make_chunk("Conclusion", "c3"),
        ]
        result = detect_repeated_headers(chunks)
        assert len(result) == 3

    def test_empty_list_returns_empty(self):
        assert detect_repeated_headers([]) == []

    def test_case_insensitive_dedup(self):
        """'INTRO' and 'intro' should be treated as the same header."""
        chunks = [
            self._make_chunk("INTRO", "c1"),
            self._make_chunk("intro", "c2"),
            self._make_chunk("Intro", "c3"),
            self._make_chunk("INTRO", "c4"),
        ]
        result = detect_repeated_headers(chunks)
        texts = [c["text"] for c in result]
        total = texts.count("INTRO") + texts.count("intro") + texts.count("Intro")
        assert total <= 3


# ══════════════════════════════════════════════════════════════════════════════
#  chunk_units (full pipeline)
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkUnits:

    def test_empty_units_list_returns_empty(self):
        result = chunk_units([], document_id="doc-1")
        assert result == []

    def test_empty_clean_text_units_are_skipped(self, sample_units):
        """Units with empty clean_text should produce zero chunks."""
        empty_only = [u for u in sample_units if not u["clean_text"].strip()]
        result = chunk_units(empty_only, document_id="doc-1")
        assert result == []

    def test_valid_units_produce_chunks(self, sample_units):
        """Valid units with text should produce at least one chunk each."""
        valid = [u for u in sample_units if u["clean_text"].strip()]
        result = chunk_units(valid, document_id="doc-1")
        assert len(result) >= len(valid)

    def test_chunk_has_required_fields(self, sample_units):
        """Every chunk must have all required fields."""
        valid  = [u for u in sample_units if u["clean_text"].strip()]
        result = chunk_units(valid, document_id="doc-1")
        required = {"chunk_id", "document_id", "unit_id", "unit_number",
                    "unit_type", "chunk_index", "text", "metadata"}
        for chunk in result:
            assert required.issubset(chunk.keys()), f"Missing keys in chunk: {chunk.keys()}"

    def test_chunk_ids_are_unique(self, sample_units):
        """Each chunk must have a unique chunk_id."""
        valid  = [u for u in sample_units if u["clean_text"].strip()]
        result = chunk_units(valid, document_id="doc-1")
        ids = [c["chunk_id"] for c in result]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_document_id_propagated(self, sample_units):
        """All chunks should carry the document_id passed in."""
        valid  = [u for u in sample_units if u["clean_text"].strip()]
        result = chunk_units(valid, document_id="my-doc-999")
        for chunk in result:
            assert chunk["document_id"] == "my-doc-999"

    def test_chunk_text_is_non_empty(self, sample_units):
        """No chunk should have empty text."""
        valid  = [u for u in sample_units if u["clean_text"].strip()]
        result = chunk_units(valid, document_id="doc-1")
        for chunk in result:
            assert chunk["text"].strip() != "", "Chunk has empty text"
