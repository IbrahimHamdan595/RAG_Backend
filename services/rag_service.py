from groq import Groq
from services.retrieval_service import search_chunk
from services.arabic_normalizer import is_arabic
from config import GROQ_API_KEY, GROQ_MODEL, MAX_CONTEXT_CHARS

client = Groq(api_key=GROQ_API_KEY)

ENGLISH_PROMPT = """You are a helpful teaching assistant.

Rules:
- Answer ONLY using the sources provided below.
- If the answer is not in the sources, say exactly: "I don't know."
- Cite sources after each fact using [Source X].
- Do NOT use outside knowledge.

Sources:
{context}

Question:
{question}

Answer:"""

ARABIC_PROMPT = """أنت مساعد تعليمي متخصص.

القواعد:
- أجب فقط باستخدام المصادر المقدمة أدناه.
- إذا لم تجد الإجابة في المصادر، قل بالضبط: "لا أعرف."
- اذكر المصدر بعد كل معلومة باستخدام [المصدر X].
- لا تستخدم معلومات خارج المصادر المقدمة.

المصادر:
{context}

السؤال:
{question}

الإجابة:"""

REFUSAL_PHRASES = ["i don't know", "i do not know", "لا أعرف", "لا اعرف", "لست متأكد"]

def _is_refusal(answer: str) -> bool:
    lower = answer.lower().strip()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def generate_response(question: str) -> dict:
    results = search_chunk(question, top_k=5)

    use_arabic = is_arabic(question)

    if not results:
        return {"answer": "لا أعرف." if use_arabic else "I don't know.", "sources": []}

    context_blocks = []
    sources        = []
    total_chars    = 0

    for i, r in enumerate(results, start=1):
        if use_arabic:
            block = f"[المصدر {i}] (وحدة {r.get('unit_number')}):\n{r['chunk']}"
        else:
            block = f"[Source {i}] (Unit {r.get('unit_number')}):\n{r['chunk']}"

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        context_blocks.append(block)
        total_chars += len(block)
        sources.append({
            "source":      f"المصدر {i}" if use_arabic else f"Source {i}",
            "unit_number": r.get("unit_number"),
            "document_id": r.get("document_id"),
            "score":       r.get("score"),
            "lang":        r.get("lang", "en"),
        })

    context  = "\n\n".join(context_blocks)
    template = ARABIC_PROMPT if use_arabic else ENGLISH_PROMPT
    prompt   = template.format(context=context, question=question)

    response = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content.strip()

    if _is_refusal(answer):
        return {"answer": "لا أعرف." if use_arabic else "I don't know.", "sources": []}

    return {"answer": answer, "sources": sources}
