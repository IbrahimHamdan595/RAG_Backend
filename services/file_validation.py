from config import ALLOWED_MINE_TYPES, MAX_FILE_SIZE_MB

def validate_file(upload_file):
	if upload_file.content_type not in ALLOWED_MINE_TYPES:
		raise ValueError("Only PDF and PPTX files are supported")

	upload_file.file.seek(0,2)
	size_mb = upload_file.file.tell() / (1024 * 1024)
	upload_file.file.seek(0)

	if size_mb > MAX_FILE_SIZE_MB:
		raise ValueError("File size exceeds limit")

	return ALLOWED_MINE_TYPES[upload_file.content_type]