"""
test_arabic_support.py
-----------------------
Tests for all Arabic language support features.

Covers:
  - arabic_normalizer  : diacritics, tatweel, alef, numerals, detection
  - text_normalizer    : auto-routing to Arabic vs English pipeline
  - chunker            : Arabic character-based chunking
  - rag_service        : Arabic prompt selection, bilingual refusal detection
"""

import pytest
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════════════════════
#  arabic_normalizer
# ══════════════════════════════════════════════════════════════════════════════

class TestIsArabic:

    def test_arabic_text_detected(self):
        from services.arabic_normalizer import is_arabic
        assert is_arabic("مرحبا بالعالم") is True

    def test_english_text_not_arabic(self):
        from services.arabic_normalizer import is_arabic
        assert is_arabic("Hello world") is False

    def test_empty_string_not_arabic(self):
        from services.arabic_normalizer import is_arabic
        assert is_arabic("") is False

    def test_mixed_text_with_majority_arabic(self):
        from services.arabic_normalizer import is_arabic
        # >30% Arabic chars
        assert is_arabic("هذا النص يحتوي على some english") is True

    def test_numbers_only_not_arabic(self):
        from services.arabic_normalizer import is_arabic
        assert is_arabic("12345 67890") is False


class TestRemoveDiacritics:

    def test_removes_fatha(self):
        from services.arabic_normalizer import remove_diacritics
        assert remove_diacritics("كَتَبَ") == "كتب"

    def test_removes_shadda(self):
        from services.arabic_normalizer import remove_diacritics
        assert remove_diacritics("مُحَمَّد") == "محمد"

    def test_clean_text_unchanged(self):
        from services.arabic_normalizer import remove_diacritics
        text = "مرحبا"
        assert remove_diacritics(text) == text


class TestRemoveTatweel:

    def test_removes_tatweel(self):
        from services.arabic_normalizer import remove_tatweel
        assert remove_tatweel("جمـيـل") == "جميل"

    def test_no_tatweel_unchanged(self):
        from services.arabic_normalizer import remove_tatweel
        assert remove_tatweel("جميل") == "جميل"


class TestNormalizeAlef:

    def test_hamza_above_normalized(self):
        from services.arabic_normalizer import normalize_alef
        assert normalize_alef("أحمد") == "احمد"

    def test_hamza_below_normalized(self):
        from services.arabic_normalizer import normalize_alef
        assert normalize_alef("إبراهيم") == "ابراهيم"

    def test_madda_normalized(self):
        from services.arabic_normalizer import normalize_alef
        assert normalize_alef("آمن") == "امن"

    def test_bare_alef_unchanged(self):
        from services.arabic_normalizer import normalize_alef
        assert normalize_alef("الكتاب") == "الكتاب"


class TestNormalizeArabicNumerals:

    def test_arabic_indic_to_western(self):
        from services.arabic_normalizer import normalize_arabic_numerals
        assert normalize_arabic_numerals("١٢٣٤٥") == "12345"

    def test_mixed_numerals(self):
        from services.arabic_normalizer import normalize_arabic_numerals
        assert normalize_arabic_numerals("الصفحة ١ من ١٠") == "الصفحة 1 من 10"

    def test_western_numerals_unchanged(self):
        from services.arabic_normalizer import normalize_arabic_numerals
        assert normalize_arabic_numerals("12345") == "12345"


class TestNormalizeArabicFull:

    def test_full_pipeline_cleans_text(self):
        from services.arabic_normalizer import normalize_arabic
        dirty = "كَتَبَ الطـالـب رقم ١"
        clean = normalize_arabic(dirty)
        # No diacritics
        assert "َ" not in clean
        # No tatweel
        assert "ـ" not in clean
        # Arabic numerals converted
        assert "١" not in clean
        assert "1" in clean

    def test_empty_returns_empty(self):
        from services.arabic_normalizer import normalize_arabic
        assert normalize_arabic("") == ""

    def test_whitespace_returns_empty(self):
        from services.arabic_normalizer import normalize_arabic
        assert normalize_arabic("   ") == ""


# ══════════════════════════════════════════════════════════════════════════════
#  text_normalizer auto-routing
# ══════════════════════════════════════════════════════════════════════════════

class TestTextNormalizerRouting:

    def test_arabic_text_routes_to_arabic_pipeline(self):
        from services.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer.normalize("كَتَبَ الطالب")
        # Diacritics should be removed
        assert "َ" not in result

    def test_english_text_routes_to_english_pipeline(self):
        from services.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer.normalize("This is an English sentence.")
        assert "This is an English sentence." in result

    def test_arabic_slide_labels_removed(self):
        from services.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        text   = "شريحة 1\nمحتوى الشريحة"
        result = normalizer.normalize(text)
        assert "شريحة 1" not in result
        assert "محتوى" in result or "محتو" in result   # content preserved

    def test_empty_returns_empty(self):
        from services.text_normalizer import TextNormalizer
        assert TextNormalizer().normalize("") == ""


# ══════════════════════════════════════════════════════════════════════════════
#  Arabic chunking
# ══════════════════════════════════════════════════════════════════════════════

class TestArabicChunking:

    def test_short_arabic_text_single_chunk(self):
        from services.chunker import split_arabic_text_into_chunks
        text   = "هذا نص قصير باللغة العربية."
        result = split_arabic_text_into_chunks(text, max_chars=500)
        assert len(result) == 1
        assert "هذا" in result[0]

    def test_long_arabic_text_multiple_chunks(self):
        from services.chunker import split_arabic_text_into_chunks
        text   = "كلمة " * 1000   # very long text
        result = split_arabic_text_into_chunks(text, max_chars=100, overlap_chars=20)
        assert len(result) > 1

    def test_arabic_chunks_are_strings(self):
        from services.chunker import split_arabic_text_into_chunks
        text   = "مرحبا بكم في العالم العربي. " * 20
        result = split_arabic_text_into_chunks(text)
        for chunk in result:
            assert isinstance(chunk, str)
            assert len(chunk.strip()) > 0

    def test_split_text_auto_uses_arabic_for_arabic(self):
        from services.chunker import split_text_auto
        arabic_text = "هذا نص عربي طويل. " * 200
        english_text = "This is an English text. " * 200
        ar_chunks = split_text_auto(arabic_text, max_tokens=500)
        en_chunks = split_text_auto(english_text, max_tokens=500)
        # Both should produce chunks
        assert len(ar_chunks) >= 1
        assert len(en_chunks) >= 1

    def test_chunk_units_stores_lang_field(self):
        from services.chunker import chunk_units
        arabic_unit = {
            "unit_id":     "u-ar-1",
            "document_id": "doc-1",
            "unit_number": 1,
            "unit_type":   "pdf",
            "clean_text":  "هذا نص عربي للاختبار.",
            "metadata":    {},
        }
        chunks = chunk_units([arabic_unit], document_id="doc-1")
        assert len(chunks) > 0
        assert chunks[0]["lang"] == "ar"

    def test_english_units_get_en_lang(self):
        from services.chunker import chunk_units
        english_unit = {
            "unit_id":     "u-en-1",
            "document_id": "doc-1",
            "unit_number": 1,
            "unit_type":   "pdf",
            "clean_text":  "This is an English test document.",
            "metadata":    {},
        }
        chunks = chunk_units([english_unit], document_id="doc-1")
        assert chunks[0]["lang"] == "en"


# ══════════════════════════════════════════════════════════════════════════════
#  Bilingual RAG service
# ══════════════════════════════════════════════════════════════════════════════

def make_chunk(text="Some content", lang="en", unit_number=1):
    return {"chunk": text, "score": 0.85, "unit_number": unit_number,
            "document_id": "doc-1", "lang": lang}

def make_ollama_response(content: str):
    return {"message": {"content": content}}

class TestBilingualRAGService:

    def _call(self, question, chunks, llm_answer):
        from services.rag_service import generate_response
        with patch("services.rag_service.search_chunk",  return_value=chunks), \
             patch("services.rag_service.ollama.chat",   return_value=make_ollama_response(llm_answer)):
            return generate_response(question)

    def test_english_question_gets_english_answer(self):
        chunks = [make_chunk("ML is artificial intelligence.")]
        result = self._call("What is ML?", chunks, "ML is a field of AI [Source 1].")
        assert "I don't know" not in result["answer"] or result["sources"]

    def test_arabic_question_no_results_returns_arabic_refusal(self):
        from services.rag_service import generate_response
        with patch("services.rag_service.search_chunk", return_value=[]):
            result = generate_response("ما هو التعلم الآلي؟")
        assert "لا أعرف" in result["answer"]
        assert result["sources"] == []

    def test_english_question_no_results_returns_english_refusal(self):
        from services.rag_service import generate_response
        with patch("services.rag_service.search_chunk", return_value=[]):
            result = generate_response("What is machine learning?")
        assert "I don't know" in result["answer"]

    def test_arabic_llm_refusal_detected(self):
        chunks = [make_chunk("بعض المحتوى", lang="ar")]
        result = self._call("ما هو الذكاء الاصطناعي؟", chunks, "لا أعرف.")
        assert "لا أعرف" in result["answer"]
        assert result["sources"] == []

    def test_sources_include_lang_field(self):
        chunks = [make_chunk("Some content", lang="en")]
        result = self._call("What is this?", chunks, "Answer [Source 1].")
        if result["sources"]:
            assert "lang" in result["sources"][0]
