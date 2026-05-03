"""
vector_store.py (replaces faiss_index.py)
------------------------------------------
Vector search backed by pgvector in Supabase instead of a local FAISS file.
Same public API — callers don't need to change.
"""

from models.model import _db


def _vec(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


def add_vectors(chunk_ids: list[str], vectors: list[list[float]]):
    with _db() as conn:
        with conn.cursor() as cur:
            for chunk_id, vector in zip(chunk_ids, vectors):
                cur.execute(
                    "UPDATE chunks SET embedding = %s::vector WHERE chunk_id = %s",
                    (_vec(vector), chunk_id),
                )


def search_vectors(query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id,
                       1 - (embedding <=> %s::vector) AS score
                FROM   chunks
                WHERE  embedding IS NOT NULL
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
                """,
                (_vec(query_vector), _vec(query_vector), top_k),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]


def remove_document_vectors(chunk_ids_to_remove: set[str]):
    if not chunk_ids_to_remove:
        return
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chunks SET embedding = NULL WHERE chunk_id = ANY(%s)",
                (list(chunk_ids_to_remove),),
            )
