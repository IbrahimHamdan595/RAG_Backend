import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# File Upload Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_docs")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

ALLOWED_MINE_TYPES = {
	"application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx"
}

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "rag_system")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Retrieval Configuration
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.4"))

# Vector Store (FAISS) Configuration
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")

# LLM Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2000"))