import uuid
import tiktoken
import re
from services.arabic_normalizer import is_arabic
from models.model import units_collection, chunks_collection

tokenizer = tiktoken.get_encoding("cl100k_base")

ARABIC_CHARS_PER_CHUNK = 2500
ARABIC_CHARS_OVERLAP   = 500


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def split_text_into_chunks(text: str, max_tokens: int = 500, overlap: int = 100) -> list[str]:
    """Token-based sliding window chunking for English text."""
    tokens = tokenizer.encode(text)
    chunks = []
    start  = 0
    while start < len(tokens):
        end         = start + max_tokens
        chunk_text  = tokenizer.decode(tokens[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks


def split_arabic_text_into_chunks(
    text: str,
    max_chars: int = ARABIC_CHARS_PER_CHUNK,
    overlap_chars: int = ARABIC_CHARS_OVERLAP
) -> list[str]:
    """Character-based sliding window chunking for Arabic text."""
    if not text:
        return []

    # BUG FIX: if text fits in one chunk, return immediately.
    # Without this, when overlap_chars >= max_chars the step becomes
    # max(max_chars - overlap_chars, 1) = 1, causing an infinite loop
    # that produces hundreds of tiny overlapping chunks.
    if len(text) <= max_chars:
        return [text.strip()]

    sentence_endings = re.compile(r'[.!?؟\n]')
    chunks = []
    start  = 0
    length = len(text)
    step   = max(max_chars - overlap_chars, 1)

    while start < length:
        end     = min(start + max_chars, length)
        segment = text[start:end]

        if end < length:
            matches = list(sentence_endings.finditer(segment))
            if matches:
                end = start + matches[-1].end()
            else:
                last_space = segment.rfind(' ')
                if last_space > max_chars // 2:
                    end = start + last_space + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def split_text_auto(text: str, max_tokens: int = 500, overlap: int = 100) -> list[str]:
    """Automatically choose chunking strategy based on language."""
    if is_arabic(text):
        return split_arabic_text_into_chunks(text, ARABIC_CHARS_PER_CHUNK, ARABIC_CHARS_OVERLAP)
    return split_text_into_chunks(text, max_tokens, overlap)


def detect_repeated_headers(chunks: list[dict]) -> list[dict]:
    seen     = {}
    filtered = []
    for chunk in chunks:
        text       = chunk["text"]
        word_count = len(text.split())
        if 0 < word_count <= 12:
            key = hash(text.strip())
            if key not in seen:
                seen[key] = 1
                filtered.append(chunk)
            else:
                seen[key] += 1
                if seen[key] <= 3:
                    filtered.append(chunk)
        else:
            filtered.append(chunk)
    return filtered


def chunk_units(units: list[dict], document_id: str,
                max_tokens: int = 500, overlap: int = 100) -> list[dict]:
    all_chunks = []

    for unit in units:
        text = unit.get("clean_text", "").strip()
        if not text:
            continue

        chunk_texts = split_text_auto(text, max_tokens, overlap)
        lang        = "ar" if is_arabic(text) else "en"

        for index, chunk_text in enumerate(chunk_texts):
            all_chunks.append({
                "chunk_id":    str(uuid.uuid4()),
                "document_id": document_id,
                "unit_id":     unit["unit_id"],
                "unit_number": unit["unit_number"],
                "unit_type":   unit["unit_type"],
                "chunk_index": index,
                "text":        chunk_text,
                "lang":        lang,
                "metadata":    unit.get("metadata", {}),
            })

    return detect_repeated_headers(all_chunks)


def chunk_units_for_document(document_id: str) -> int:
    units  = list(units_collection.find({"document_id": document_id}))
    chunks = chunk_units(units, document_id)
    if chunks:
        chunks_collection.insert_many(chunks)
    return len(chunks)