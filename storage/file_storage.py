import os
from uuid import uuid4
from config import UPLOAD_DIR

def store_file(upload_file, extension):
	os.makedirs(UPLOAD_DIR, exist_ok=True)

	file_id = str(uuid4())
	file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{extension}")

	with open(file_path, "wb") as f:
		f.write(upload_file.file.read())

		return file_path