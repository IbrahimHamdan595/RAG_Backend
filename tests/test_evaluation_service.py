"""
test_evaluation_service.py
---------------------------
Tests for services/evaluation_service.py

Key things we verify:
  - The return-inside-loop bug is FIXED (returns all questions, not just the first)
  - Correct refusal is detected correctly
  - Grounded answers are detected correctly
  - Mixed batches work end-to-end
"""

import pytest
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_rag_response(answer: str, sources: list = None):
    """Build a fake RAG response dict."""
    return {"answer": answer, "sources": sources or []}


# ══════════════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluateQuestions:

    def _run(self, questions, fake_responses):
        """
        Helper: patch generate_response to return fake_responses in sequence,
        then call evaluate_questions and return the report.
        """
        from services.evaluation_service import evaluate_questions

        with patch("services.evaluation_service.generate_response",
                   side_effect=fake_responses):
            return evaluate_questions(questions)

    # ── The bug fix ──────────────────────────────────────────────────────────

    def test_returns_all_questions_not_just_first(self):
        """
        THE KEY BUG FIX TEST.
        Before the fix, return was inside the for-loop → only 1 result returned.
        After the fix, all questions must appear in the report.
        """
        questions = [
            {"question": "Q1", "expected_answer": "some answer"},
            {"question": "Q2", "expected_answer": "another answer"},
            {"question": "Q3", "expected_answer": "I don't know"},
        ]
        fake_responses = [
            make_rag_response("Some answer about Q1", [{"source": "Source 1", "unit_number": 1, "document_id": "d", "score": 0.9}]),
            make_rag_response("Some answer about Q2", [{"source": "Source 1", "unit_number": 2, "document_id": "d", "score": 0.8}]),
            make_rag_response("I don't know.", []),
        ]

        report = self._run(questions, fake_responses)

        # MUST return 3 results, not 1
        assert len(report) == 3, (
            f"Expected 3 results but got {len(report)}. "
            "This is the return-inside-loop bug — check evaluation_service.py"
        )

    # ── Correct refusal ──────────────────────────────────────────────────────

    def test_correct_refusal_passes(self):
        """Model says 'I don't know' and expected is also 'I don't know' → passed=True."""
        questions = [{"question": "What is 42?", "expected_answer": "I don't know"}]
        responses = [make_rag_response("I don't know.", [])]

        report = self._run(questions, responses)
        assert report[0]["passed"] is True

    def test_wrong_refusal_fails(self):
        """Model says 'I don't know' but expected is a real answer → passed=False."""
        questions = [{"question": "What is ML?", "expected_answer": "machine learning"}]
        responses = [make_rag_response("I don't know.", [])]

        report = self._run(questions, responses)
        assert report[0]["passed"] is False

    # ── Grounded answers ─────────────────────────────────────────────────────

    def test_grounded_answer_passes(self):
        """Model gives an answer with sources → grounded → passed=True."""
        questions = [{"question": "What is ML?", "expected_answer": "machine learning"}]
        sources   = [{"source": "Source 1", "unit_number": 1, "document_id": "d", "score": 0.9}]
        responses = [make_rag_response("ML is a field of AI [Source 1].", sources)]

        report = self._run(questions, responses)
        assert report[0]["passed"] is True

    def test_answer_without_sources_fails(self):
        """Model gives an answer but no sources → not grounded → passed=False."""
        questions = [{"question": "What is ML?", "expected_answer": "machine learning"}]
        responses = [make_rag_response("ML is great.", [])]  # no sources!

        report = self._run(questions, responses)
        assert report[0]["passed"] is False

    # ── Report fields ────────────────────────────────────────────────────────

    def test_report_contains_required_fields(self):
        """Every report entry must have: question, expected, answer, sources_count, passed."""
        questions = [{"question": "Test?", "expected_answer": "I don't know"}]
        responses = [make_rag_response("I don't know.", [])]

        report = self._run(questions, responses)
        required = {"question", "expected", "answer", "sources_count", "passed"}
        assert required.issubset(report[0].keys())

    def test_sources_count_is_correct(self):
        """sources_count in the report should match the actual number of sources returned."""
        questions = [{"question": "Q?", "expected_answer": "something"}]
        sources   = [
            {"source": "Source 1", "unit_number": 1, "document_id": "d", "score": 0.9},
            {"source": "Source 2", "unit_number": 2, "document_id": "d", "score": 0.7},
        ]
        responses = [make_rag_response("Answer with two sources.", sources)]

        report = self._run(questions, responses)
        assert report[0]["sources_count"] == 2

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_empty_questions_list_returns_empty_report(self):
        from services.evaluation_service import evaluate_questions
        with patch("services.evaluation_service.generate_response"):
            report = evaluate_questions([])
        assert report == []

    def test_case_insensitive_dont_know_detection(self):
        """'I DON'T KNOW' and 'i don't know' should both be detected as refusal."""
        questions = [{"question": "Q?", "expected_answer": "I don't know"}]
        responses = [make_rag_response("I DON'T KNOW.", [])]

        report = self._run(questions, responses)
        assert report[0]["passed"] is True
