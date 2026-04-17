from sentence_transformers import SentenceTransformer
from models.model import chunks_collection
from services.faiss_index import search_vectors
from config import EMBEDDING_MODEL, MIN_SIMILARITY_SCORE

model = SentenceTransformer(EMBEDDING_MODEL)

def search_chunk(query: str, top_k: int = 5) -> list[dict]:
    """
    Search for the most relevant chunks using FAISS vector search.

    Flow:
      1. Encode the query into a vector
      2. Ask FAISS for top-K nearest chunk_ids (fast — no Python loop)
      3. Fetch the actual chunk text from MongoDB by chunk_id
      4. Filter by minimum similarity score

    Returns list of dicts with: chunk, score, unit_number, document_id
    """

    # 1️⃣ Encode query
    query_vector = model.encode(query).tolist()

    # 2️⃣ Search FAISS for top-K chunk_ids
    faiss_results = search_vectors(query_vector, top_k=top_k)

    if not faiss_results:
        return []

    # 3️⃣ Filter by minimum score threshold
    faiss_results = [
        (chunk_id, score)
        for chunk_id, score in faiss_results
        if score >= MIN_SIMILARITY_SCORE
    ]

    if not faiss_results:
        return []

    # 4️⃣ Fetch chunk metadata from MongoDB using chunk_ids
    chunk_id_list = [chunk_id for chunk_id, _ in faiss_results]
    score_map     = {chunk_id: score for chunk_id, score in faiss_results}

    mongo_chunks = chunks_collection.find({"chunk_id": {"$in": chunk_id_list}})

    # Build a lookup dict so we preserve FAISS ranking order
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in mongo_chunks}

    # 5️⃣ Assemble results in FAISS score order
    results = []
    for chunk_id in chunk_id_list:
        chunk = chunk_lookup.get(chunk_id)
        if not chunk:
            continue

        results.append({
            "chunk":       chunk["text"],
            "score":       score_map[chunk_id],
            "unit_number": chunk.get("unit_number"),
            "document_id": chunk.get("document_id"),
            "lang":        chunk.get("lang", "en"),
        })

    return results
