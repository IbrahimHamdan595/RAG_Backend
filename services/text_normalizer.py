"""
Unified text normalizer for both Arabic and English content.

Automatically detects the language of each unit and applies the
appropriate cleaning pipeline:
  - Arabic text  → arabic_normalizer.normalize_arabic()
  - English text → existing regex-based cleaning
  - Mixed text   → Arabic pipeline (preserves both scripts)
"""

import re
from services.arabic_normalizer import normalize_arabic, is_arabic


class TextNormalizer:
    """
    Stateless normalizer — create once, call .normalize() on each unit.
    """
    def normalize(self, text: str) -> str:
        """
        Auto-detect language and apply the right cleaning pipeline.
        Returns cleaned text ready for chunking and embedding.
        """
        if not text or not text.strip():
            return ""

        if is_arabic(text):
            return self._normalize_arabic(text)
        else:
            return self._normalize_english(text)

    def _normalize_arabic(self, text: str) -> str:
        """
        Arabic pipeline:
          1. Apply full Arabic normalization (diacritics, tatweel, alef, etc.)
          2. Remove slide/page labels (in Arabic too)
          3. Remove footers and standalone numbers
          4. Deduplicate lines
        """
        # Apply Arabic-specific normalization first
        text = normalize_arabic(text)

        # Remove slide/page labels (Arabic + English variants)
        text = re.sub(r'^(شريحة|صفحة|Slide|Page)\s*\d+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove standalone numbers (page numbers, slide numbers)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # Remove copyright footers
        text = re.sub(r'(©|حقوق النشر|جميع الحقوق).*$', '', text, flags=re.MULTILINE)

        # Deduplicate lines (RTL-safe: compare stripped lowercase)
        seen  = set()
        lines = []
        for line in text.splitlines():
            key = line.strip()
            if key and key not in seen:
                seen.add(key)
                lines.append(line)

        text = "\n".join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _normalize_english(self, text: str) -> str:
        """
        Original English pipeline — unchanged from the existing implementation.
        """
        text = text.replace("\r\n", "\n")

        # Remove slide/page labels
        text = re.sub(r'^Slide\s+\d+.*$', '', text, flags=re.MULTILINE)

        # Remove course headers (e.g. CS101)
        text = re.sub(r'^[A-Z]{2,}\d{2,}.*$', '', text, flags=re.MULTILINE)

        # Remove standalone numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # Remove copyright footers
        text = re.sub(r'©.*$', '', text, flags=re.MULTILINE)

        # Deduplicate lines
        seen  = set()
        lines = []
        for line in text.splitlines():
            key = line.strip().lower()
            if key and key not in seen:
                seen.add(key)
                lines.append(line)

        text = "\n".join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

def normalize_text(text: str) -> str:
    """Backwards-compatible module-level function."""
    return TextNormalizer().normalize(text)