from fastapi import APIRouter
from services.rag_service import generate_response

router = APIRouter()

@router.post("/ask")
def ask(question: dict):
	return generate_response(question['question'])