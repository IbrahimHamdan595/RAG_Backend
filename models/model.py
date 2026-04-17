from datetime import datetime
from uuid import uuid4
try:
    from pymongo import MongoClient
except ImportError as e:
    raise RuntimeError(
        "Missing dependency 'pymongo'. Install it with: python -m pip install -r requirements.txt"
    ) from e

import config

client = MongoClient(config.MONGO_URI)
db = client[config.MONGO_DB_NAME]
documents_collection = db["documents"]
units_collection = db["units"]
chunks_collection = db["chunks"]

def create_document_record(file_name, file_type, storage_path):
	document = {
        "document_id": str(uuid4()),
        "file_name": file_name,
        "file_type": file_type,  # 'pdf' or 'pptx'
        "uploaded_at": datetime.utcnow(),
        "status": "uploaded",     # uploaded | processed | failed
        "storage_path": storage_path,
        "total_units": None       # total pages/slides, to fill later
    }
	documents_collection.insert_one(document)
	return document

def create_unit_record(document_id, unit_number, raw_text, unit_type="page", title=None, clean_text=None, metadata=None):
    unit = {
        "unit_id": str(uuid4()),
        "document_id": document_id,
        "unit_type": unit_type,  # 'page' or 'slide'
        "unit_number": unit_number,
        "title": title,
        "raw_text": raw_text,
        "clean_text":  clean_text,
        "metadata":    metadata or {"unit_number": unit_number, "unit_type": unit_type},
        "created_at": datetime.utcnow()
    }
    units_collection.insert_one(unit)
    return unit

def create_chunk_record(document_id, unit_id, chunk_index, chunk_text, token_count, overlap_from_previous=False):
    chunk = {
        "chunk_id": str(uuid4()),
        "document_id": document_id,
        "unit_id": unit_id,
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "token_count": token_count,
        "metadata": {
            "overlap_from_previous": overlap_from_previous
        },
        "created_at": datetime.utcnow()
    }
    chunks_collection.insert_one(chunk)
    return chunk

def update_document_units(document_id, total_units):
    documents_collection.update_one(
        {"document_id": document_id},
        {"$set": {
            "total_units": total_units,
            "status": "processed"
        }}
    )