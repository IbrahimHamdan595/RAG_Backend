from fastapi import APIRouter, HTTPException
from models.model import documents_collection, update_document_units
from services.pptx_ingestion import ingest_pptx

router = APIRouter()

@router.post("/ingest/pptx/{document_id}")
def ingest_pptx_endpoint(document_id: str):
    document = documents_collection.find_one({"document_id": document_id})

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document["file_type"] != "pptx":
        raise HTTPException(status_code=400, detail="Not a PPTX document")

    total_slides = ingest_pptx(document)
    update_document_units(document_id, total_slides)

    return {
        "document_id": document_id,
        "slides_ingested": total_slides
    }
