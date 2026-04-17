from services.rag_service import generate_response

REFUSAL_PHRASES = ["i don't know", "i do not know", "لا أعرف", "لا اعرف"]

def _is_refusal(text: str) -> bool:
    lower = text.lower().strip()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)

def evaluate_questions(questions: list):
	report = []
	for q in questions:
		response = generate_response(q["question"])

		expected_is_refusal = _is_refusal(q["expected_answer"])
		answer_is_refusal   = _is_refusal(response["answer"])

		correct_refusal = expected_is_refusal and answer_is_refusal

		grounded = (
            not answer_is_refusal and
            len(response["sources"]) > 0
        )

		report.append({
			"question": q["question"],
			"expected": q["expected_answer"],
			"answer": response["answer"],
			"sources_count": len(response["sources"]),
			"passed": correct_refusal or grounded,
		})

	return report