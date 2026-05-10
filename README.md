# 🎓 EduEval AI — Academic Answer Evaluation System

> Powered by **RAG · Pinecone · Supabase · Groq LLaMA-3 · Sentence Transformers · Streamlit**

Fully cloud-native: vector index persists in **Pinecone**, evaluation history persists in **Supabase**.  
No local files — survives every redeploy.

---

## 🔑 You Need 4 Free API Keys

| Service | Free Tier | Get Key |
|---------|-----------|---------|
| **Groq** | Unlimited (rate-limited) | [console.groq.com](https://console.groq.com) |
| **Pinecone** | 1 index · 100K vectors | [app.pinecone.io](https://app.pinecone.io) |
| **Supabase** | 500 MB DB · unlimited calls | [supabase.com](https://supabase.com) |

---

## 🗄️ One-Time Supabase Setup

After creating your Supabase project, open the **SQL Editor** and run this once:

```sql
CREATE TABLE IF NOT EXISTS evaluations (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    student_name      TEXT        DEFAULT 'Anonymous',
    question          TEXT        NOT NULL,
    student_answer    TEXT        NOT NULL,
    marks_awarded     INTEGER     NOT NULL,
    max_marks         INTEGER     NOT NULL,
    percentage        FLOAT       NOT NULL,
    grade             TEXT        NOT NULL,
    concepts_covered  JSONB,
    concepts_missing  JSONB,
    strengths         JSONB,
    weaknesses        JSONB,
    detailed_feedback TEXT,
    improved_answer   TEXT,
    context_used      TEXT
);
```

---

## 🚀 Deploy to Streamlit Cloud (Free)

### 1 — Push to GitHub
```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/edueval-ai.git
git push -u origin main
```

### 2 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Select your repo, set main file to `app.py`
3. Click **Advanced settings → Secrets** and paste:

```toml
GROQ_API_KEY     = "gsk_your_key"
PINECONE_API_KEY = "pcsk_your_key"
SUPABASE_URL     = "https://xxxx.supabase.co"
SUPABASE_KEY     = "eyJ_your_anon_key"
```

4. Click **Deploy** — live in ~2 minutes ✅

---

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your 4 keys
streamlit run app.py
```

---

## 📁 Project Structure

```
edueval-ai/
├── app.py              ← Main Streamlit UI (4 tabs)
├── rag_engine.py       ← Pinecone vector DB + RAG retrieval
├── evaluator.py        ← Groq LLM evaluation logic
├── database.py         ← Supabase evaluation history
├── prompts.py          ← Prompt builder utility
├── requirements.txt    ← Python dependencies
├── packages.txt        ← System deps for Streamlit Cloud
├── .env.example        ← Local secrets template
└── .streamlit/
    ├── config.toml     ← Theme + server config
    └── secrets.toml    ← Local secrets (never commit!)
```

---

## 🏗️ Architecture

```
OFFLINE (one-time setup):
  Syllabus PDFs → PyMuPDF → 500-char chunks
       → Sentence Transformer embeddings (384-dim)
       → Pinecone upsert (persists forever ☁️)

ONLINE (per evaluation):
  Question + Student Answer
        ↓
  Embed → Pinecone cosine search → Top-5 chunks
        ↓
  Prompt = [System] + [Question] + [Answer] + [Context]
        ↓
  Groq LLaMA-3.3 70B → structured JSON
        ↓
  Supabase INSERT (persists forever ☁️)
        ↓
  Marks | Grade | Concepts | Feedback | Model Answer
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Persistent Vector DB** | Pinecone — PDFs indexed once, available across all deployments |
| **Persistent History** | Supabase — every evaluation stored, never lost on redeploy |
| **Semantic Retrieval** | Cosine similarity search over syllabus chunks |
| **AI Evaluation** | Groq LLaMA-3.3 70B scores concept coverage & correctness |
| **Explainable Results** | Covered/missing concepts, strengths, weaknesses, model answer |
| **Batch Mode** | Evaluate entire class from a CSV |
| **Grade Analytics** | Charts + stats across all evaluations |
| **CSV Export** | Download full history any time |
