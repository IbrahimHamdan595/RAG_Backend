from fastapi import APIRouter, HTTPException
from models.model import documents_collection
from services.chunker import chunk_units_for_document

router = APIRouter()

@router.post("/chunk/{document_id}")
def chunk_document_units(document_id: str):
	document = documents_collection.find_one({"document_id": document_id})
	if not document:
		raise HTTPException(status_code=404, detail="Document not found")

	chunked_count = chunk_units_for_document(document_id)

	return {
		"document_id": document_id,
		"chunked_count": chunked_count
	}