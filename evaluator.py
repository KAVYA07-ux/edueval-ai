"""
Evaluator — sends question + student answer + RAG context to the Groq LLM
and parses the structured evaluation response.
"""

import json
import re
from groq import Groq

# ── Model config ─────────────────────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"   # Free, fast, high quality
MAX_TOKENS   = 1500
TEMPERATURE  = 0.2                  # Low = more consistent scoring


# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert academic evaluator.
Your job is to evaluate a student's descriptive answer against reference material retrieved from the course syllabus/textbook.

You MUST respond with ONLY a valid JSON object — no markdown fences, no explanation outside JSON.

JSON schema (follow exactly):
{
  "marks_awarded": <integer 0-10>,
  "max_marks": 10,
  "percentage": <float>,
  "grade": <"A+" | "A" | "B" | "C" | "D" | "F">,
  "concepts_covered": [<string>, ...],
  "concepts_missing": [<string>, ...],
  "strengths": [<string>, ...],
  "weaknesses": [<string>, ...],
  "detailed_feedback": "<2-4 sentences of constructive feedback>",
  "improved_answer": "<a model answer the student can learn from, 3-6 sentences>"
}

Grading scale:
  9-10 → A+,  8 → A,  7 → B,  5-6 → C,  3-4 → D,  0-2 → F
"""

USER_PROMPT_TEMPLATE = """QUESTION:
{question}

STUDENT'S ANSWER:
{student_answer}

REFERENCE MATERIAL (from course syllabus/textbook):
{context}

Evaluate the student's answer strictly based on the reference material above.
Award marks out of 10 based on concept coverage, correctness, and clarity.
Return ONLY the JSON object."""


class Evaluator:
    """Wraps the Groq API and handles prompt construction + response parsing."""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def evaluate(self, question: str, student_answer: str, context: str) -> dict:
        """
        Run the evaluation.  Returns a parsed dict on success,
        or a dict with 'error' key on failure.
        """
        user_msg = USER_PROMPT_TEMPLATE.format(
            question=question,
            student_answer=student_answer,
            context=context if context else "No reference material available. Evaluate based on general knowledge.",
        )

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_text = response.choices[0].message.content.strip()
            return self._parse_response(raw_text)

        except Exception as e:
            return {"error": str(e)}

    # ── Response parsing ─────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> dict:
        """Extract JSON from the LLM response robustly."""
        # Strip markdown fences if the model added them anyway
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

        try:
            data = json.loads(text)
            return self._validate(data)
        except json.JSONDecodeError:
            # Try to find JSON block inside the text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return self._validate(data)
                except json.JSONDecodeError:
                    pass
            return {"error": f"Could not parse LLM response:\n{text}"}

    def _validate(self, data: dict) -> dict:
        """Ensure required keys exist; fill defaults for optional ones."""
        required = ["marks_awarded", "grade", "concepts_covered",
                    "concepts_missing", "detailed_feedback", "improved_answer"]
        for key in required:
            if key not in data:
                data[key] = "N/A"

        # Ensure numeric fields are correct types
        if "marks_awarded" in data:
            try:
                data["marks_awarded"] = int(data["marks_awarded"])
            except (ValueError, TypeError):
                data["marks_awarded"] = 0

        if "percentage" not in data or data["percentage"] == "N/A":
            try:
                data["percentage"] = round(data["marks_awarded"] / 10 * 100, 1)
            except Exception:
                data["percentage"] = 0.0

        data.setdefault("max_marks", 10)
        data.setdefault("strengths",  [])
        data.setdefault("weaknesses", [])
        return data
