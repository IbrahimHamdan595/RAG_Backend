"""
Generates vector embeddings using a multilingual sentence-transformer model.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - Supports 50+ languages including Arabic, English, French, etc.
  - Same 384-dim output as all-MiniLM-L6-v2 → FAISS index stays compatible
  - Slightly slower than the English-only model but handles Arabic natively
  - Cross-lingual: an Arabic question can retrieve English chunks and vice versa

Why not AraBERT?
  AraBERT is Arabic-only. The multilingual model lets users ask questions
  in either language and retrieve chunks from documents in either language.
"""

from sentence_transformers import SentenceTransformer
from models.model import chunks_collection
from services.faiss_index import add_vectors
from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def generate_embedding(document_id: str):
	chunks = list(chunks_collection.find({"document_id": document_id}))

	chunck_ids = []
	vectors = []

	for chunk in chunks:

		text = chunk.get("text", "").strip()
		if not text:
			continue

		vector = model.encode(text).tolist()

		chunks_collection.update_one(
			{"_id": chunk["_id"]},
			{"$set": {"embedding": vector}}
		)

		chunck_ids.append(chunk["chunk_id"])
		vectors.append(vector)

	# 2️⃣ Bulk-add all vectors to FAISS in one call (much faster than one-by-one)
	if chunck_ids:
		add_vectors(chunck_ids, vectors)

	return len(chunck_ids)