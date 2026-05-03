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

# Database Configuration (Supabase / PostgreSQL)
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Embedding Configuration (HuggingFace Inference API)
HF_API_TOKEN    = os.getenv("HF_API_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Retrieval Configuration
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.4"))

# Vector Store (FAISS) Configuration
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")

# LLM Configuration (Groq)
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2000"))