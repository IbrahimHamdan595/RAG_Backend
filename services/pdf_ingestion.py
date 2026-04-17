from services.pdf_extractor import extract_pdf_pages
from services.text_normalizer import TextNormalizer
from services.chunker import chunk_units
from models.model import create_unit_record, units_collection, chunks_collection


def ingest_pdf(document):
    normalizer = TextNormalizer()

    pages = extract_pdf_pages(document["storage_path"])

    # 1. Create units
    for page in pages:
        clean_text = normalizer.normalize(page["text"])

        create_unit_record(
            document_id=document["document_id"],
            unit_number=page["page_number"],
            raw_text=page["text"],
            clean_text=clean_text,
            unit_type="page",
            metadata={
                "source_type": "pdf",
                "page_number": page["page_number"],
                "slide_number": None
            }
        )

    # 2. Fetch units
    units = list(units_collection.find({
        "document_id": document["document_id"]
    }))

    # 3. Prevent duplicate chunking
    if chunks_collection.count_documents({"document_id": document["document_id"]}) == 0:
        chunks = chunk_units(
            units=units,
            document_id=document["document_id"]
        )

        if chunks:
            chunks_collection.insert_many(chunks)

    return len(pages)
