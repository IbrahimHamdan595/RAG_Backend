import numpy as np
from huggingface_hub import InferenceClient
from models.model import chunks_collection
from services.faiss_index import add_vectors
from config import EMBEDDING_MODEL, HF_API_TOKEN

_client = InferenceClient(token=HF_API_TOKEN)


def _embed(text: str) -> list[float]:
    result = _client.feature_extraction(text, model=EMBEDDING_MODEL)
    arr = np.asarray(result)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr.tolist()


def generate_embedding(document_id: str) -> int:
    chunks = chunks_collection.find({"document_id": document_id})

    chunk_ids = []
    vectors   = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        vector = _embed(text)

        chunks_collection.update_one(
            {"chunk_id": chunk["chunk_id"]},
            {"$set": {"embedding": vector}}
        )

        chunk_ids.append(chunk["chunk_id"])
        vectors.append(vector)

    if chunk_ids:
        add_vectors(chunk_ids, vectors)

    return len(chunk_ids)
