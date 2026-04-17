"""
conftest.py — shared pytest fixtures
-------------------------------------
Fixtures here are available to every test file automatically.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# ── Make sure the project root is on sys.path ─────────────────────────────────
# Tests live in /tests/, project code lives one level up in /RAG Claude/
# Adjust this path to match your local folder name if different.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── Patch MongoDB BEFORE any project module is imported ──────────────────────
# This prevents tests from needing a real MongoDB connection.
# Any test that needs a real DB should use the `real_db` marker instead.

@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    """
    Replace MongoClient with a MagicMock for every test by default.
    This means no test needs a real running MongoDB.
    """
    mock_client   = MagicMock()
    mock_db       = MagicMock()
    mock_client.__getitem__.return_value = mock_db

    with patch("pymongo.MongoClient", return_value=mock_client):
        yield mock_client


@pytest.fixture
def sample_units():
    """A list of clean units as produced by the ingestion pipeline."""
    return [
        {
            "unit_id":     "unit-001",
            "document_id": "doc-001",
            "unit_number": 1,
            "unit_type":   "pdf",
            "clean_text":  "Machine learning is a subset of artificial intelligence. "
                           "It enables systems to learn from data without being explicitly programmed. "
                           "Common algorithms include decision trees, neural networks, and SVMs.",
            "metadata":    {"unit_number": 1, "unit_type": "pdf"},
        },
        {
            "unit_id":     "unit-002",
            "document_id": "doc-001",
            "unit_number": 2,
            "unit_type":   "pdf",
            "clean_text":  "Deep learning uses multi-layer neural networks to model complex patterns. "
                           "It has achieved state-of-the-art results in computer vision and NLP tasks. "
                           "Training requires large datasets and significant compute.",
            "metadata":    {"unit_number": 2, "unit_type": "pdf"},
        },
        {
            "unit_id":     "unit-003",
            "document_id": "doc-001",
            "unit_number": 3,
            "unit_type":   "pdf",
            "clean_text":  "",   # ← empty unit — should be skipped by chunker
            "metadata":    {"unit_number": 3, "unit_type": "pdf"},
        },
    ]


@pytest.fixture
def sample_chunks():
    """A list of chunks as produced by the chunking pipeline."""
    return [
        {
            "chunk_id":    "chunk-001",
            "document_id": "doc-001",
            "unit_id":     "unit-001",
            "unit_number": 1,
            "unit_type":   "pdf",
            "chunk_index": 0,
            "text":        "Machine learning is a subset of artificial intelligence.",
            "metadata":    {},
        },
        {
            "chunk_id":    "chunk-002",
            "document_id": "doc-001",
            "unit_id":     "unit-002",
            "unit_number": 2,
            "unit_type":   "pdf",
            "chunk_index": 0,
            "text":        "Deep learning uses multi-layer neural networks.",
            "metadata":    {},
        },
    ]
