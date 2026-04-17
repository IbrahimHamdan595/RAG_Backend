from services.pptx_extractor import extract_pptx_slides
from services.text_normalizer import TextNormalizer
from services.chunker import chunk_units
from models.model import create_unit_record, units_collection, chunks_collection


def ingest_pptx(document):
    normalizer = TextNormalizer()

    slides = extract_pptx_slides(document["storage_path"])

    # 1. Create units
    for slide in slides:
        clean_text = normalizer.normalize(slide["text"])

        create_unit_record(
            document_id=document["document_id"],
            unit_number=slide["slide_number"],
            raw_text=slide["text"],
            clean_text=clean_text,
            unit_type="slide",
            title=slide["title"],
            metadata={
                "source_type": "pptx",
                "page_number": None,
                "slide_number": slide["slide_number"]
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

    return len(slides)
