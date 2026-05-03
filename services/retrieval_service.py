import numpy as np
from huggingface_hub import InferenceClient
from models.model import chunks_collection
from services.faiss_index import search_vectors
from config import EMBEDDING_MODEL, HF_API_TOKEN, MIN_SIMILARITY_SCORE

_client = InferenceClient(token=HF_API_TOKEN)


def _embed(text: str) -> list[float]:
    result = _client.feature_extraction(text, model=EMBEDDING_MODEL)
    arr = np.asarray(result)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr.tolist()


def search_chunk(query: str, top_k: int = 5) -> list[dict]:
    query_vector = _embed(query)

    results = search_vectors(query_vector, top_k=top_k)

    results = [(cid, score) for cid, score in results if score >= MIN_SIMILARITY_SCORE]
    if not results:
        return []

    chunk_id_list = [cid for cid, _ in results]
    score_map     = {cid: score for cid, score in results}

    chunks = chunks_collection.find({"chunk_id": {"$in": chunk_id_list}})
    lookup = {c["chunk_id"]: c for c in chunks}

    return [
        {
            "chunk":       lookup[cid]["text"],
            "score":       score_map[cid],
            "unit_number": lookup[cid].get("unit_number"),
            "document_id": lookup[cid].get("document_id"),
            "lang":        lookup[cid].get("lang", "en"),
        }
        for cid in chunk_id_list
        if cid in lookup
    ]
