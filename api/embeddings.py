from fastapi import APIRouter
from services.embedding_service import generate_embedding

router = APIRouter()

@router.post("/embed/{document_id}")
def embed_document(document_id: str):
	embedding_result = generate_embedding(document_id)
	return {
		"document_id": document_id,
		"chunks_embedded": embedding_result
	}