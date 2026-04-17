from fastapi import APIRouter
from services.evaluation_service import evaluate_questions

router = APIRouter()

@router.post("/evaluate")
def evaluate(payload: dict):
	return evaluate_questions(payload["questions"])