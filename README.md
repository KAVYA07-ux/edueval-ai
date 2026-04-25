<<<<<<< HEAD
# 🎓 AI-Based Academic Answer Evaluation System

> Powered by **RAG · FAISS · Groq LLaMA-3 · Sentence Transformers · Streamlit**

---

## 🚀 Quick Start (5 Minutes)

### Step 1 — Get a FREE Groq API Key
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up (free) → API Keys → Create Key
3. Copy the key

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Set Your API Key
```bash
# Option A: Create a .env file
cp .env.example .env
# Edit .env and paste your key

# Option B: Just paste it in the app's sidebar (no file needed)
```

### Step 4 — Run the App
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
answer_evaluator/
│
├── app.py              ← Main Streamlit UI
├── rag_engine.py       ← FAISS vector DB + RAG retrieval
├── evaluator.py        ← Groq LLM evaluation logic
├── requirements.txt    ← All dependencies
├── .env.example        ← API key template
│
├── uploaded_pdfs/      ← Auto-created when you upload PDFs
└── vector_store/       ← Auto-created FAISS index + metadata
    ├── faiss.index
    └── metadata.pkl
```

---

## 🏗️ System Architecture

```
OFFLINE (one-time setup):
  Syllabus PDFs → Text Extraction → Chunking → Sentence Embeddings → FAISS Index

ONLINE (per evaluation):
  Question + Student Answer
        ↓
  Embed Question → FAISS Search → Top-K Relevant Chunks
        ↓
  Prompt = [System Prompt] + [Question] + [Answer] + [Retrieved Context]
        ↓
  Groq LLaMA-3 70B
        ↓
  Marks | Grade | Concepts Covered/Missing | Feedback | Model Answer
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **PDF Knowledge Base** | Upload any syllabus or textbook PDF |
| **Semantic Retrieval** | FAISS finds the most relevant reference chunks |
| **AI Evaluation** | LLM scores based on concept coverage & correctness |
| **Explainable Results** | Covered concepts, missing concepts, strengths, weaknesses |
| **Model Answer** | Suggested improved answer for student learning |
| **Batch Mode** | Evaluate entire class via CSV upload |
| **Persistent Index** | FAISS index saved to disk — survives app restarts |

---

## 📊 How Evaluation Works

The LLM receives:
1. The **question**
2. The **student's answer**
3. **Reference context** retrieved from your uploaded PDFs via semantic search

It then returns a structured JSON with:
- `marks_awarded` (out of 10)
- `grade` (A+ to F)
- `concepts_covered` — what the student got right
- `concepts_missing` — gaps in the answer
- `strengths` — positives
- `weaknesses` — areas to improve
- `detailed_feedback` — 2-4 sentence constructive feedback
- `improved_answer` — a model answer the student can learn from

---

## 🛠️ Technologies Used

| Technology | Version | Role |
|-----------|---------|------|
| Python | 3.10+ | Core language |
| Streamlit | 1.35 | Web UI |
| Groq API (LLaMA-3 70B) | latest | LLM evaluation |
| FAISS | 1.8 | Vector similarity search |
| Sentence Transformers | 3.0 | Text embeddings |
| PyMuPDF | 1.24 | PDF text extraction |
| LangChain Text Splitters | 0.2 | Intelligent text chunking |
| Plotly | 5.22 | Score visualization |

---

## 💡 Tips for Best Results

1. **Upload relevant PDFs** — the more specific your syllabus material, the better the evaluation
2. **Clear questions** — specific questions yield better context retrieval
3. **Chunk quality** — the system uses 500-char chunks with 100-char overlap for optimal retrieval

---

## 📝 Sample CSV Format (Batch Mode)

```csv
student_name,question,student_answer
Alice,What is RAG?,RAG stands for Retrieval Augmented Generation...
Bob,Explain FAISS,FAISS is a library for efficient similarity search...
```
=======
Emerging tools Lab Work
>>>>>>> 1a2b3350ade651a153a44dbc49d1cb094333612d
