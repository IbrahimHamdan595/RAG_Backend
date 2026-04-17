"""
arabic_normalizer.py
---------------------
Arabic-specific text normalization for RAG preprocessing.

Why Arabic needs special treatment:
  - Diacritics (tashkeel / harakat): أَ أُ أِ — same word, different marks
    → Strip them so "كَتَبَ" and "كتب" match the same chunk
  - Tatweel (ـ): decorative elongation e.g. "جمـيـل" → "جميل"
    → Remove it, adds no semantic value
  - Alef variants: أ إ آ ا — all mean the same base letter
    → Normalize to bare ا so queries match correctly
  - Waw/Ya variants: ؤ → و  and  ئ → ي
  - Ta marbuta: ة → ه (optional, helps matching across dialects)
  - Arabic-Indic numerals: ١٢٣ → 123
  - Repeated characters: "جمييييل" → "جميل"
  - Zero-width characters (ZWJ, ZWNJ) that appear in copied Arabic text
"""
import re
import unicodedata

# unicode ranges for Arabic characters
ARABIC_RANGE       = r'\u0600-\u06FF'    # Arabic block
ARABIC_SUPPLEMENT  = r'\u0750-\u077F'    # Arabic Supplement
ARABIC_EXTENDED_A  = r'\u08A0-\u08FF'    # Arabic Extended-A
ARABIC_PRES_A      = r'\uFB50-\uFDFF'    # Arabic Presentation Forms-A
ARABIC_PRES_B      = r'\uFE70-\uFEFF'    # Arabic Presentation Forms-B

# Regex to detect if a text is predominantly Arabic
_ARABIC_CHAR_RE = re.compile(f'[{ARABIC_RANGE}]')

def is_arabic(text: str, threshold: float = 0.3) -> bool:
	"""
    Return True if at least `threshold` fraction of the characters are Arabic.
    A 30% threshold handles mixed Arabic/English documents.
    """
	if not text:
		return False
	arabic_chars = len(_ARABIC_CHAR_RE.findall(text))
	return (arabic_chars / max(len(text), 1)) >= threshold

def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel/harakat).
    These are the short vowel marks placed above/below letters.
    Unicode range: U+064B–U+065F
    """
    return re.sub(r'[\u064B-\u065F\u0670]', '', text)

def remove_tatweel(text: str) -> str:
    """Remove tatweel (kashida / ـ) — decorative letter extension."""
    return text.replace('\u0640', '')

def normalize_alef(text: str) -> str:
    """
    Normalize all Alef variants to bare Alef (ا).
    Variants: أ (U+0623), إ (U+0625), آ (U+0622), أ with hamza above/below
    """
    text = re.sub(r'[أإآٱ]', 'ا', text)
    return text

def normalize_hamza(text: str) -> str:
    """
    Normalize hamza variants:
      ؤ (U+0624) → و
      ئ (U+0626) → ي
      ء (U+0621) — kept as is (standalone hamza)
    """
    text = text.replace('\u0624', '\u0648')  # ؤ → و
    text = text.replace('\u0626', '\u064A')  # ئ → ي
    return text

def normalize_ta_marbuta(text: str) -> str:
    """
    Normalize ta marbuta (ة U+0629) → ha (ه U+0647).
    Helps match "مدرسة" and "مدرسه" as the same word.
    """
    return text.replace('\u0629', '\u0647')

def normalize_arabic_numerals(text: str) -> str:
    """
    Convert Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) to Western numerals (0123456789).
    Also converts Persian numerals (۰۱۲۳۴۵۶۷۸۹).
    """
    arabic_indic = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    persian      = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
    return text.translate(arabic_indic).translate(persian)

def remove_zero_width(text: str) -> str:
    """Remove zero-width characters that appear in copy-pasted Arabic text."""
    # ZWJ (U+200D), ZWNJ (U+200C), ZWS (U+200B), BOM (U+FEFF)
    return re.sub(r'[\u200B-\u200D\uFEFF\u200F\u200E]', '', text)

def collapse_repeated_arabic_chars(text: str) -> str:
    """
    Collapse elongated/repeated Arabic characters.
    e.g. "جمييييل" → "جميل"  |  "ههههه" → "هه" (keep max 2)
    """
    return re.sub(r'([\u0600-\u06FF])\1{2,}', r'\1\1', text)

def normalize_arabic_punctuation(text: str) -> str:
    """
    Normalize Arabic punctuation to ASCII equivalents:
      ، → ,   (Arabic comma)
      ؛ → ;   (Arabic semicolon)
      ؟ → ?   (Arabic question mark)
    """
    text = text.replace('\u060C', ',')
    text = text.replace('\u061B', ';')
    text = text.replace('\u061F', '?')
    return text

def normalize_arabic(text: str) -> str:
    """
    Full Arabic normalization pipeline. Apply all steps in order.

    Input:  raw Arabic text (possibly with diacritics, tatweel, etc.)
    Output: clean, normalized Arabic text suitable for embedding
    """
    if not text:
        return ""

    text = remove_zero_width(text)
    text = remove_diacritics(text)
    text = remove_tatweel(text)
    text = normalize_alef(text)
    text = normalize_hamza(text)
    text = normalize_ta_marbuta(text)
    text = normalize_arabic_numerals(text)
    text = normalize_arabic_punctuation(text)
    text = collapse_repeated_arabic_chars(text)

    # Normalize unicode (NFC form — consistent character composition)
    text = unicodedata.normalize('NFC', text)

    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()