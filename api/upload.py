from fastapi import APIRouter, UploadFile, File, HTTPException
from services.file_validation import validate_file
from storage.file_storage import store_file
from models.model import create_document_record

router = APIRouter()

@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
	try:
		file_type = validate_file(file)
		storage_path = store_file(file, file_type)
		document = create_document_record(
			file_name = file.filename,
			file_type = file_type,
			storage_path = storage_path
		)

		return {
			"document_id": document["document_id"],
			"status": "uploaded"
		}
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
