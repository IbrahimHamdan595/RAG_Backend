from fastapi import APIRouter
from services.retrieval_service import search_chunk

router = APIRouter()

@router.post("/search")
def search(query: str, top_k: int = 5):
	results = search_chunk(query, top_k)
	return {"results": results}