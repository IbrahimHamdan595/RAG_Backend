"""
faiss_index.py
--------------
Manages the FAISS vector index and the mapping between
FAISS integer IDs  ↔  MongoDB chunk_ids.

Why a separate file?
  - FAISS uses integer IDs internally (0, 1, 2 ...)
  - MongoDB uses string UUIDs ("abc-123-...")
  - This module bridges the two so nothing else has to care about it.

Files saved to disk:
  - vector_store/faiss.index   → the FAISS index binary
  - vector_store/id_map.json   → list of chunk_ids in FAISS order
"""

import os
import json
import numpy as np
import faiss

# ─── Paths ───────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
ID_MAP_PATH      = os.path.join(VECTOR_STORE_DIR, "id_map.json")

# ─── Embedding dimension for all-MiniLM-L6-v2 ────────────────────────────────
EMBEDDING_DIM = 384


def _ensure_dir():
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


def load_index() -> tuple[faiss.Index, list[str]]:
    """
    Load the FAISS index and chunk_id map from disk.
    Creates empty ones if they don't exist yet.

    Returns:
        index   — FAISS IndexFlatIP (inner product = cosine on normalized vecs)
        id_map  — list of chunk_ids; position == FAISS integer ID
    """
    _ensure_dir()

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ID_MAP_PATH):
        index  = faiss.read_index(FAISS_INDEX_PATH)
        with open(ID_MAP_PATH, "r") as f:
            id_map = json.load(f)
    else:
        # IndexFlatIP = exact inner-product search
        # We normalize vectors before adding, so inner product == cosine similarity
        index  = faiss.IndexFlatIP(EMBEDDING_DIM)
        id_map = []

    return index, id_map


def save_index(index: faiss.Index, id_map: list[str]):
    """Persist the FAISS index and id_map to disk."""
    _ensure_dir()
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(ID_MAP_PATH, "w") as f:
        json.dump(id_map, f)


def add_vectors(chunk_ids: list[str], vectors: list[list[float]]):
    """
    Add new vectors to the FAISS index.

    Args:
        chunk_ids — MongoDB chunk_ids matching each vector
        vectors   — raw embeddings (will be L2-normalized before storing)
    """
    index, id_map = load_index()

    matrix = np.array(vectors, dtype="float32")

    # Normalize → cosine similarity becomes inner product
    faiss.normalize_L2(matrix)

    index.add(matrix)
    id_map.extend(chunk_ids)

    save_index(index, id_map)


def search_vectors(query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
    """
    Search the FAISS index for the top-K most similar chunks.

    Args:
        query_vector — raw query embedding (will be normalized)
        top_k        — number of results to return

    Returns:
        List of (chunk_id, score) tuples, sorted by score descending
    """
    index, id_map = load_index()

    if index.ntotal == 0:
        return []

    query = np.array([query_vector], dtype="float32")
    faiss.normalize_L2(query)

    # D = distances (scores), I = integer indices in FAISS
    scores, indices = index.search(query, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:           # FAISS returns -1 for empty slots
            continue
        chunk_id = id_map[idx]
        results.append((chunk_id, float(score)))

    return results


def remove_document_vectors(chunk_ids_to_remove: set[str]):
    """
    Remove all vectors belonging to a specific document.
    FAISS doesn't support deletion, so we rebuild the index
    without the removed chunk_ids.

    Args:
        chunk_ids_to_remove — set of chunk_ids to exclude
    """
    index, id_map = load_index()

    if index.ntotal == 0:
        return

    # Reconstruct all existing vectors
    all_vectors = index.reconstruct_n(0, index.ntotal)

    # Filter out the ones we want to remove
    kept_ids     = []
    kept_vectors = []

    for i, chunk_id in enumerate(id_map):
        if chunk_id not in chunk_ids_to_remove:
            kept_ids.append(chunk_id)
            kept_vectors.append(all_vectors[i])

    # Rebuild fresh index
    new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    if kept_vectors:
        matrix = np.array(kept_vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        new_index.add(matrix)

    save_index(new_index, kept_ids)