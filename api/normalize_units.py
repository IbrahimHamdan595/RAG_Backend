from fastapi import APIRouter, HTTPException
from models.model import documents_collection
from services.unit_normalization import normalize_units_for_document

router = APIRouter()

@router.post("/normalize/{document_id}")
def normalize_document_units(document_id: str):
    document = documents_collection.find_one({"document_id": document_id})

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    normalized_count = normalize_units_for_document(document_id)

    return {
        "document_id": document_id,
        "units_normalized": normalized_count
    }
