def build_evaluation_prompt(question, student_answer, context):
    prompt = f"""
You are an academic evaluator.

Reference Material:
{context}

Question:
{question}

Student Answer:
{student_answer}

Tasks:
1. Identify covered concepts
2. Identify missing concepts
3. Evaluate correctness
4. Provide constructive feedback
5. Suggest improvements

Important:
Only use the reference material for evaluation.
Do not use outside knowledge.
"""
    return prompt
