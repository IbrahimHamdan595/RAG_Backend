"""
test_faiss_index.py
--------------------
Tests for faiss_index.py

What we test:
  - Adding vectors and searching returns correct results
  - Scores are within valid cosine range [-1, 1]
  - Searching empty index returns empty list
  - Removing document vectors works correctly
  - Index persists to disk and reloads correctly
"""

import pytest
import numpy as np
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_vector_store(tmp_path):
    """
    Redirect FAISS file paths to a temp directory so tests don't
    pollute the real vector_store/ folder and are fully isolated.
    """
    with patch("services.faiss_index.VECTOR_STORE_DIR", str(tmp_path)), \
         patch("services.faiss_index.FAISS_INDEX_PATH", str(tmp_path / "faiss.index")), \
         patch("services.faiss_index.ID_MAP_PATH",      str(tmp_path / "id_map.json")):
        yield tmp_path

def random_vector(dim=384):
    """Return a random unit vector of given dimension."""
    v = np.random.randn(dim).astype("float32")
    return (v / np.linalg.norm(v)).tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFaissIndex:

    def test_search_empty_index_returns_empty(self, tmp_vector_store):
        from faiss_index import search_vectors
        results = search_vectors(random_vector(), top_k=5)
        assert results == []

    def test_add_and_search_returns_results(self, tmp_vector_store):
        from faiss_index import add_vectors, search_vectors

        vec = random_vector()
        add_vectors(["chunk-001"], [vec])

        results = search_vectors(vec, top_k=1)
        assert len(results) == 1
        chunk_id, score = results[0]
        assert chunk_id == "chunk-001"

    def test_exact_match_has_high_score(self, tmp_vector_store):
        """Searching with the exact same vector should return score ≈ 1.0."""
        from faiss_index import add_vectors, search_vectors

        vec = random_vector()
        add_vectors(["chunk-exact"], [vec])

        results = search_vectors(vec, top_k=1)
        _, score = results[0]
        assert score > 0.99, f"Expected score near 1.0, got {score}"

    def test_scores_are_in_valid_range(self, tmp_vector_store):
        """All returned scores should be in the cosine similarity range [-1, 1]."""
        from faiss_index import add_vectors, search_vectors

        vecs = [random_vector() for _ in range(10)]
        ids  = [f"chunk-{i}" for i in range(10)]
        add_vectors(ids, vecs)

        results = search_vectors(random_vector(), top_k=10)
        for _, score in results:
            assert -1.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_top_k_limits_results(self, tmp_vector_store):
        """search_vectors(top_k=3) should return at most 3 results."""
        from faiss_index import add_vectors, search_vectors

        vecs = [random_vector() for _ in range(10)]
        ids  = [f"chunk-{i}" for i in range(10)]
        add_vectors(ids, vecs)

        results = search_vectors(random_vector(), top_k=3)
        assert len(results) <= 3

    def test_results_are_sorted_by_score_descending(self, tmp_vector_store):
        """Results should be ordered from highest to lowest score."""
        from faiss_index import add_vectors, search_vectors

        vecs = [random_vector() for _ in range(5)]
        ids  = [f"chunk-{i}" for i in range(5)]
        add_vectors(ids, vecs)

        results = search_vectors(random_vector(), top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    def test_index_persists_to_disk_and_reloads(self, tmp_vector_store):
        """Vectors added in one call should be findable after reloading from disk."""
        from faiss_index import add_vectors, search_vectors, load_index

        vec = random_vector()
        add_vectors(["persistent-chunk"], [vec])

        # Force reload from disk by calling load_index again
        index, id_map = load_index()
        assert "persistent-chunk" in id_map

        # Search should still work
        results = search_vectors(vec, top_k=1)
        assert results[0][0] == "persistent-chunk"

    def test_multiple_adds_accumulate(self, tmp_vector_store):
        """Calling add_vectors multiple times should accumulate, not replace."""
        from faiss_index import add_vectors, search_vectors

        add_vectors(["chunk-A"], [random_vector()])
        add_vectors(["chunk-B"], [random_vector()])
        add_vectors(["chunk-C"], [random_vector()])

        _, id_map = __import__("faiss_index").load_index()
        assert len(id_map) == 3

    def test_remove_document_vectors(self, tmp_vector_store):
        """After removing chunk IDs, they should no longer appear in search results."""
        from faiss_index import add_vectors, search_vectors, remove_document_vectors

        vec_keep   = random_vector()
        vec_remove = random_vector()

        add_vectors(["keep-me"],    [vec_keep])
        add_vectors(["remove-me"],  [vec_remove])

        # Remove the second chunk
        remove_document_vectors({"remove-me"})

        _, id_map = __import__("faiss_index").load_index()
        assert "remove-me" not in id_map
        assert "keep-me"   in id_map

    def test_remove_from_empty_index_does_not_crash(self, tmp_vector_store):
        """Removing from an empty index should silently do nothing."""
        from faiss_index import remove_document_vectors
        remove_document_vectors({"nonexistent-chunk"})  # should not raise
