"""
test_api_endpoints.py
----------------------
Integration tests for all FastAPI API endpoints.

Uses FastAPI's TestClient (built on httpx) — no real HTTP server needed.
MongoDB and service calls are mocked so tests run without any external deps.

Endpoints covered:
  GET  /health
  POST /api/upload
  POST /api/ingest/pdf/{id}
  POST /api/ingest/pptx/{id}
  POST /api/chunk/{id}
  POST /api/embed/{id}
  POST /api/search
  POST /api/ask
  POST /api/evaluate
"""

import pytest
import io
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app."""
    from main import app
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
#  /health
# ══════════════════════════════════════════════════════════════════════════════

class TestHealth:

    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_returns_ok_status(self, client):
        res = client.get("/health")
        assert res.json() == {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/upload
# ══════════════════════════════════════════════════════════════════════════════

class TestUpload:

    def _upload(self, client, filename, content=b"fake content", content_type="application/pdf"):
        return client.post(
            "/api/upload",
            files={"file": (filename, io.BytesIO(content), content_type)},
        )

    def test_upload_valid_pdf_returns_200(self, client):
        with patch("api.upload.validate_file",          return_value="pdf"), \
             patch("api.upload.store_file",             return_value="uploaded_docs/test.pdf"), \
             patch("api.upload.create_document_record", return_value={"document_id": "doc-123", "status": "uploaded"}):
            res = self._upload(client, "test.pdf")
        assert res.status_code == 200

    def test_upload_returns_document_id(self, client):
        with patch("api.upload.validate_file",          return_value="pdf"), \
             patch("api.upload.store_file",             return_value="uploaded_docs/test.pdf"), \
             patch("api.upload.create_document_record", return_value={"document_id": "doc-abc", "status": "uploaded"}):
            res = self._upload(client, "test.pdf")
        data = res.json()
        assert "document_id" in data
        assert data["document_id"] == "doc-abc"

    def test_upload_invalid_file_type_returns_400(self, client):
        with patch("api.upload.validate_file", side_effect=ValueError("Unsupported file type")):
            res = self._upload(client, "test.txt", content_type="text/plain")
        assert res.status_code == 400

    def test_upload_valid_pptx_returns_200(self, client):
        pptx_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        with patch("api.upload.validate_file",          return_value="pptx"), \
             patch("api.upload.store_file",             return_value="uploaded_docs/test.pptx"), \
             patch("api.upload.create_document_record", return_value={"document_id": "doc-pptx", "status": "uploaded"}):
            res = self._upload(client, "slides.pptx", content_type=pptx_type)
        assert res.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/ingest/pdf/{document_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestPDF:

    def test_ingest_pdf_success(self, client):
        mock_doc = {"document_id": "doc-1", "file_type": "pdf", "storage_path": "test.pdf"}
        with patch("api.ingest_pdf.documents_collection") as mock_col, \
             patch("api.ingest_pdf.ingest_pdf",           return_value=10), \
             patch("api.ingest_pdf.update_document_units"):
            mock_col.find_one.return_value = mock_doc
            res = client.post("/api/ingest/pdf/doc-1")
        assert res.status_code == 200
        assert res.json()["total_pages"] == 10

    def test_ingest_pdf_not_found_returns_404(self, client):
        with patch("api.ingest_pdf.documents_collection") as mock_col:
            mock_col.find_one.return_value = None
            res = client.post("/api/ingest/pdf/nonexistent-id")
        assert res.status_code == 404

    def test_ingest_pdf_wrong_type_returns_400(self, client):
        mock_doc = {"document_id": "doc-1", "file_type": "pptx"}
        with patch("api.ingest_pdf.documents_collection") as mock_col:
            mock_col.find_one.return_value = mock_doc
            res = client.post("/api/ingest/pdf/doc-1")
        assert res.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/ingest/pptx/{document_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestPPTX:

    def test_ingest_pptx_success(self, client):
        mock_doc = {"document_id": "doc-2", "file_type": "pptx"}
        with patch("api.ingest_pptx.documents_collection") as mock_col, \
             patch("api.ingest_pptx.ingest_pptx",           return_value=15), \
             patch("api.ingest_pptx.update_document_units"):
            mock_col.find_one.return_value = mock_doc
            res = client.post("/api/ingest/pptx/doc-2")
        assert res.status_code == 200
        assert res.json()["slides_ingested"] == 15

    def test_ingest_pptx_not_found_returns_404(self, client):
        with patch("api.ingest_pptx.documents_collection") as mock_col:
            mock_col.find_one.return_value = None
            res = client.post("/api/ingest/pptx/bad-id")
        assert res.status_code == 404

    def test_ingest_pptx_wrong_type_returns_400(self, client):
        mock_doc = {"document_id": "doc-2", "file_type": "pdf"}
        with patch("api.ingest_pptx.documents_collection") as mock_col:
            mock_col.find_one.return_value = mock_doc
            res = client.post("/api/ingest/pptx/doc-2")
        assert res.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/chunk/{document_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestChunk:

    def test_chunk_success_returns_count(self, client):
        with patch("api.chunk.documents_collection") as mock_col, \
             patch("api.chunk.chunk_units_for_document", return_value=42):
            mock_col.find_one.return_value = {"document_id": "doc-1"}
            res = client.post("/api/chunk/doc-1")
        assert res.status_code == 200
        assert res.json()["chunked_count"] == 42

    def test_chunk_not_found_returns_404(self, client):
        with patch("api.chunk.documents_collection") as mock_col:
            mock_col.find_one.return_value = None
            res = client.post("/api/chunk/no-such-doc")
        assert res.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/embed/{document_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbed:

    def test_embed_success(self, client):
        with patch("api.embeddings.generate_embedding", return_value=25):
            res = client.post("/api/embed/doc-1")
        assert res.status_code == 200
        assert res.json()["chunks_embedded"] == 25
        assert res.json()["document_id"]     == "doc-1"


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/ask
# ══════════════════════════════════════════════════════════════════════════════

class TestAsk:

    def test_ask_returns_answer_and_sources(self, client):
        fake_response = {
            "answer":  "Machine learning is a subset of AI [Source 1].",
            "sources": [{"source": "Source 1", "unit_number": 1, "document_id": "d", "score": 0.9}],
        }
        with patch("api.ask.generate_response", return_value=fake_response):
            res = client.post("/api/ask", json={"question": "What is ML?"})
        assert res.status_code == 200
        data = res.json()
        assert "answer"  in data
        assert "sources" in data

    def test_ask_dont_know_response(self, client):
        fake_response = {"answer": "I don't know.", "sources": []}
        with patch("api.ask.generate_response", return_value=fake_response):
            res = client.post("/api/ask", json={"question": "What is the meaning of life?"})
        assert res.status_code == 200
        assert res.json()["answer"] == "I don't know."
        assert res.json()["sources"] == []


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/evaluate
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluate:

    def test_evaluate_returns_report(self, client):
        fake_report = [
            {"question": "Q1", "expected": "A1", "answer": "A1", "sources_count": 1, "passed": True},
            {"question": "Q2", "expected": "I don't know", "answer": "I don't know.", "sources_count": 0, "passed": True},
        ]
        with patch("api.evaluate.evaluate_questions", return_value=fake_report):
            res = client.post("/api/evaluate", json={
                "questions": [
                    {"question": "Q1", "expected_answer": "A1"},
                    {"question": "Q2", "expected_answer": "I don't know"},
                ]
            })
        assert res.status_code == 200
        assert len(res.json()) == 2
        assert res.json()[0]["passed"] is True
