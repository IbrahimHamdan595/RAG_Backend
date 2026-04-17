"""
pdf_extractor.py
-----------------
Extracts text from PDF files with OCR fallback for image-based pages.

Two-pass strategy:
  Pass 1: pdfplumber — fast, accurate for text-layer PDFs
  Pass 2: pytesseract OCR — for scanned/image pages that have no text layer

Why OCR fallback matters:
  Many academic lecture PDFs are scanned images. Without OCR,
  pdfplumber returns empty strings for those pages, making them
  invisible to the RAG pipeline entirely.

Requirements for OCR:
  pip install pytesseract pdf2image pillow
  + Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki (Windows)
  + Poppler binary:   https://github.com/oschwartz10612/poppler-windows (Windows)
"""

import pdfplumber

# OCR imports — optional, gracefully disabled if not installed
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# Minimum characters to consider a page as "has text" — below this we try OCR
MIN_TEXT_CHARS = 20


def _extract_with_ocr(file_path: str, page_number: int) -> str:
    """
    Convert a single PDF page to an image and run Tesseract OCR on it.
    Returns extracted text or empty string if OCR fails.
    """
    try:
        images = convert_from_path(
            file_path,
            first_page=page_number,
            last_page=page_number,
            dpi=200   # 200 DPI is a good balance of speed vs accuracy
        )
        if not images:
            return ""

        # Run OCR — 'ara+eng' handles mixed Arabic/English documents
        text = pytesseract.image_to_string(images[0], lang="ara+eng")
        return text.strip()

    except Exception:
        # If OCR fails for any reason, return empty — don't crash the pipeline
        return ""


def extract_pdf_pages(file_path: str) -> list[dict]:
    """
    Extract text from all pages of a PDF.

    For each page:
      - If pdfplumber finds text → use it (fast)
      - If page is blank/image AND OCR is available → run Tesseract
      - If OCR is not installed → skip the page (log a warning)

    Returns list of dicts with: source_type, page_number, slide_number, text
    """
    pages = []

    with pdfplumber.open(file_path) as pdf:
        total = len(pdf.pages)

        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            # Page has no text layer — try OCR
            if len(text) < MIN_TEXT_CHARS:
                if OCR_AVAILABLE:
                    text = _extract_with_ocr(file_path, i)
                else:
                    # OCR not installed — skip silently
                    # To enable: pip install pytesseract pdf2image pillow
                    pass

            pages.append({
                "source_type":  "pdf",
                "page_number":  i,
                "slide_number": None,
                "text":         text,
                "ocr_used":     len((page.extract_text() or "").strip()) < MIN_TEXT_CHARS
            })

    return pages