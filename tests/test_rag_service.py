"""
test_rag_service.py
--------------------
Tests for services/rag_service.py

What we test:
  - Returns "I don't know" when no chunks are retrieved
  - Returns "I don't know" when LLM says it doesn't know
  - Returns answer + sources when LLM gives a grounded response
  - Context is correctly capped at MAX_CONTEXT_CHARS
  - Sources list matches the retrieved chunks
"""

import pytest
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_chunk(unit_number=1, doc_id="doc-1", score=0.85, text="Some relevant chunk text."):
    return {
        "chunk":       text,
        "score":       score,
        "unit_number": unit_number,
        "document_id": doc_id,
    }

def make_ollama_response(content: str):
    """Fake ollama.chat() response structure."""
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: {"message": {"content": content}}[key]
    return {"message": {"content": content}}


# ══════════════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateResponse:

    def _call(self, search_results, llm_answer="A grounded answer [Source 1]."):
        """
        Patch search_chunk and ollama.chat, then call generate_response.
        Returns the result dict.
        """
        from services.rag_service import generate_response

        with patch("services.rag_service.search_chunk",  return_value=search_results), \
             patch("services.rag_service.ollama.chat",    return_value=make_ollama_response(llm_answer)):
            return generate_response("What is machine learning?")

    # ── No retrieval results ─────────────────────────────────────────────────

    def test_no_chunks_returns_dont_know(self):
        """When FAISS returns nothing, we should immediately say 'I don't know'."""
        result = self._call(search_results=[])
        assert result["answer"] == "I don't know."
        assert result["sources"] == []

    # ── LLM refuses ─────────────────────────────────────────────────────────

    def test_llm_says_dont_know_returns_dont_know(self):
        """When the LLM says 'I don't know', we return a clean refusal."""
        chunks = [make_chunk()]
        result = self._call(search_results=chunks, llm_answer="I don't know.")
        assert result["answer"] == "I don't know."
        assert result["sources"] == []

    def test_llm_says_dont_know_case_insensitive(self):
        """The refusal check should be case-insensitive."""
        chunks = [make_chunk()]
        result = self._call(search_results=chunks, llm_answer="I DON'T KNOW anything about this.")
        assert result["answer"] == "I don't know."

    # ── Successful answer ────────────────────────────────────────────────────

    def test_grounded_answer_returned_with_sources(self):
        """A proper LLM answer should be returned along with source metadata."""
        chunks = [make_chunk(unit_number=3, score=0.91)]
        result = self._call(search_results=chunks, llm_answer="ML is a subset of AI [Source 1].")
        assert "ML is a subset of AI" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["sources"][0]["unit_number"] == 3
        assert result["sources"][0]["score"] == 0.91

    def test_sources_count_matches_chunks_count(self):
        """Number of sources in response should match retrieved chunks (up to context limit)."""
        chunks = [make_chunk(unit_number=i, score=0.9 - i * 0.1) for i in range(1, 4)]
        result = self._call(search_results=chunks, llm_answer="Answer based on sources [Source 1].")
        # All 3 chunks are short, so all 3 should appear as sources
        assert len(result["sources"]) == 3

    def test_source_fields_are_present(self):
        """Each source must have: source, unit_number, document_id, score."""
        chunks = [make_chunk()]
        result = self._call(search_results=chunks, llm_answer="Some answer [Source 1].")
        required = {"source", "unit_number", "document_id", "score"}
        assert required.issubset(result["sources"][0].keys())

    # ── Context cap ──────────────────────────────────────────────────────────

    def test_context_is_capped_at_max_chars(self):
        """
        When chunks together exceed MAX_CONTEXT_CHARS, only enough chunks
        to fill the context window should be included as sources.
        """
        from services import rag_service

        # Temporarily set a very small context limit
        original = rag_service.MAX_CONTEXT_CHARS
        rag_service.MAX_CONTEXT_CHARS = 50  # very small — forces truncation

        try:
            # Create chunks whose text is each > 50 chars
            chunks = [make_chunk(unit_number=i, text="A" * 60) for i in range(5)]
            result = self._call(search_results=chunks, llm_answer="Answer [Source 1].")
            # Should have fewer than 5 sources because context was capped
            assert len(result["sources"]) < 5
        finally:
            rag_service.MAX_CONTEXT_CHARS = original
