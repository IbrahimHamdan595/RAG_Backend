from models.model import units_collection
from services.text_normalizer import normalize_text

def normalize_units_for_document(document_id):
    units = units_collection.find({"document_id": document_id})
    modified = 0

    for unit in units:
        raw = unit.get("raw_text", "")
        clean = normalize_text(raw)

        if clean != unit.get("clean_text"):
            units_collection.update_one(
                {"unit_id": unit["unit_id"]},
                {"$set": {"clean_text": clean}}
            )
            modified += 1

    return modified