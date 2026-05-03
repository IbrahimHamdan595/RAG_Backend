from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.upload import router as upload_router
from api.ingest_pdf import router as ingest_pdf_router
from api.ingest_pptx import router as pptx_ingest_router
from api.normalize_units import router as normalize_router
from api.chunk import router as chunk_router
from api.embeddings import router as embeddings_router
from api.search import router as search_router
from api.ask import router as ask_router
from api.evaluate import router as evaluate_router

app = FastAPI(title="RAG Ingestion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ──────────────────────────────────────────────────────────────
# Required by Docker HEALTHCHECK — returns 200 when the server is up
@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(upload_router, prefix="/api")
app.include_router(ingest_pdf_router, prefix="/api")
app.include_router(pptx_ingest_router, prefix="/api")
app.include_router(normalize_router, prefix="/api")
app.include_router(chunk_router, prefix="/api")
app.include_router(embeddings_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(ask_router, prefix="/api")
app.include_router(evaluate_router, prefix="/api")