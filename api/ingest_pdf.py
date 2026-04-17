from fastapi import APIRouter, HTTPException
from models.model import documents_collection, update_document_units
from services.pdf_ingestion import ingest_pdf

router = APIRouter()

@router.post("/ingest/pdf/{document_id}")
def ingest_pdf_endpoint(document_id: str):
	document = documents_collection.find_one({"document_id": document_id})

	if not document:
		raise HTTPException(status_code=404, detail="Document not found or not a PDF")

	if document["file_type"] != "pdf":
		raise HTTPException(status_code=400, detail="Document is not a PDF")

	total_pages = ingest_pdf(document)
	update_document_units(document_id, total_pages)

	return {
		"document_id": document_id,
		"total_pages": total_pages
	}