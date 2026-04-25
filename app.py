"""
AI-Based Academic Answer Evaluation System
Streamlit UI — Redesigned with premium look
"""

import os
import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

from rag_engine import RAGEngine
from evaluator  import Evaluator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduEval AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0a0a0f; color: #e8e8f0; }

[data-testid="stSidebar"] { background: #0f0f1a !important; border-right: 1px solid #1e1e30; }
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }

.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(88,166,255,0.06) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(163,113,247,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #58a6ff 0%, #a371f7 50%, #f778ba 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0 !important;
}
.hero p { color: #8b949e !important; font-size: 0.95rem; margin: 0; }
.hero-tags { display: flex; gap: 8px; margin-top: 1rem; flex-wrap: wrap; }
.hero-tag {
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.2);
    color: #58a6ff !important;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 500; letter-spacing: 0.5px;
}

.card { background: #0d1117; border: 1px solid #21262d; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: #58a6ff; margin-bottom: 0.75rem;
}

.metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 1.5rem; }
.metric-box { background: #0d1117; border: 1px solid #21262d; border-radius: 10px; padding: 1rem 1.2rem; text-align: center; }
.metric-box .label { font-size: 0.7rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.metric-box .value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #e8e8f0; line-height: 1; }

.chips-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.chip { padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 500; }
.chip-green { background: rgba(35,134,54,0.15); border: 1px solid rgba(35,134,54,0.3); color: #3fb950; }
.chip-red   { background: rgba(248,81,73,0.1);  border: 1px solid rgba(248,81,73,0.2);  color: #f85149; }

.fb-box {
    background: #0d1117; border: 1px solid #21262d;
    border-left: 3px solid #58a6ff; border-radius: 8px;
    padding: 1rem 1.2rem; margin-top: 8px;
    font-size: 0.9rem; line-height: 1.6; color: #c8c8d8;
}
.fb-box.green { border-left-color: #3fb950; }

.sw-item { display: flex; align-items: flex-start; gap: 8px; padding: 6px 0; font-size: 0.88rem; color: #c8c8d8; border-bottom: 1px solid #161b22; }
.sw-item:last-child { border-bottom: none; }
.sw-dot { width: 6px; height: 6px; border-radius: 50%; margin-top: 6px; flex-shrink: 0; }
.dot-green { background: #3fb950; }
.dot-red   { background: #f85149; }

[data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #21262d; }
[data-baseweb="tab"] { color: #8b949e !important; }
[aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

.stTextArea textarea, .stTextInput input {
    background: #0d1117 !important; border: 1px solid #30363d !important;
    border-radius: 8px !important; color: #e8e8f0 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    rag = RAGEngine()
    rag.load_index()
    st.session_state.rag = rag
if "history" not in st.session_state:
    st.session_state.history = []

GRADE_COLORS = {"A+": "#3fb950","A": "#3fb950","B": "#58a6ff","C": "#e3b341","D": "#f0883e","F": "#f85149"}

def grade_color(g): return GRADE_COLORS.get(g, "#8b949e")

def marks_to_grade(marks, max_marks):
    pct = marks / max_marks * 100
    if pct >= 90: return "A+"
    if pct >= 80: return "A"
    if pct >= 70: return "B"
    if pct >= 50: return "C"
    if pct >= 30: return "D"
    return "F"

def score_gauge(marks, max_marks=10):
    color = grade_color(marks_to_grade(marks, max_marks))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=marks,
        number={"suffix": f"/{max_marks}", "font": {"size": 40, "color": "#e8e8f0", "family": "Syne"}},
        gauge={
            "axis": {"range": [0, max_marks], "tickcolor": "#30363d", "tickwidth": 1, "tickfont": {"color": "#8b949e", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0d1117", "bordercolor": "#21262d",
            "steps": [
                {"range": [0, max_marks*0.3],  "color": "rgba(248,81,73,0.08)"},
                {"range": [max_marks*0.3, max_marks*0.5], "color": "rgba(240,136,62,0.08)"},
                {"range": [max_marks*0.5, max_marks*0.7], "color": "rgba(227,179,65,0.08)"},
                {"range": [max_marks*0.7, max_marks*0.9], "color": "rgba(88,166,255,0.08)"},
                {"range": [max_marks*0.9, max_marks],     "color": "rgba(63,185,80,0.08)"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(height=220, margin=dict(t=30,b=10,l=30,r=30), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font={"color": "#e8e8f0"})
    return fig

def chips_html(items, cls):
    if not items: return "<span style='color:#8b949e;font-size:0.85rem'>None identified</span>"
    return "".join(f'<span class="chip {cls}">{i}</span>' for i in items)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='padding:1rem 0 0.5rem;'><div style='font-family:Syne;font-size:1.1rem;font-weight:800;background:linear-gradient(135deg,#58a6ff,#a371f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>EduEval AI</div><div style='font-size:0.75rem;color:#8b949e;'>Academic Answer Evaluator</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""), placeholder="gsk_...")
    if api_key: st.success("✓ API key loaded")
    else:
        st.warning("Enter your Groq API key")
        st.markdown("[Get free key →](https://console.groq.com)")

    st.markdown("---")
    st.markdown('<div class="section-label">Knowledge Base</div>', unsafe_allow_html=True)
    rag: RAGEngine = st.session_state.rag
    ca, cb = st.columns(2)
    ca.metric("Chunks", rag.total_chunks)
    cb.metric("Status", "Ready" if rag.total_chunks > 0 else "Empty")
    uploaded = st.file_uploader("Upload syllabus PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        if st.button("Index PDFs", use_container_width=True, type="primary"):
            prog = st.progress(0)
            paths = []
            os.makedirs("uploaded_pdfs", exist_ok=True)
            for i, f in enumerate(uploaded):
                dest = f"uploaded_pdfs/{f.name}"
                with open(dest, "wb") as out: out.write(f.read())
                paths.append(dest)
                prog.progress((i+1)/len(uploaded)*0.4)
            with st.spinner("Generating embeddings…"):
                n = rag.add_documents(paths)
                prog.progress(1.0)
            st.success(f"Indexed {n} chunks!")
            st.rerun()
    if rag.total_chunks > 0:
        if st.button("Clear Knowledge Base", use_container_width=True):
            rag.clear_index(); st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label">Recent Evaluations</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            g = h["grade"]
            st.markdown(f"<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #21262d;font-size:0.82rem;'><span style='color:#c8c8d8;'>{h['question'][:28]}…</span><span style='color:{grade_color(g)};font-weight:700;'>{g}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#8b949e;font-size:0.85rem'>No evaluations yet</span>", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎓 EduEval AI</h1>
    <p>Intelligent academic answer evaluation powered by Retrieval-Augmented Generation</p>
    <div class="hero-tags">
        <span class="hero-tag">RAG Pipeline</span>
        <span class="hero-tag">FAISS Vector DB</span>
        <span class="hero-tag">Groq LLaMA-3</span>
        <span class="hero-tag">Sentence Transformers</span>
        <span class="hero-tag">Explainable AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  📝  Evaluate Answer  ", "  📦  Batch Evaluation  ", "  🏗️  Architecture  "])

with tab1:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        question = st.text_area("Question", placeholder="e.g. Explain Retrieval-Augmented Generation and its applications.", height=110)
        answer   = st.text_area("Student's Answer", placeholder="Paste the student's descriptive answer here…", height=200)
        max_marks = st.select_slider("Max Marks", options=[5, 10, 15, 20], value=10)
        evaluate  = st.button("🚀  Evaluate Answer", use_container_width=True, type="primary", disabled=not api_key)

    with right:
        st.markdown('<div class="section-label">Retrieved Context</div>', unsafe_allow_html=True)
        if question:
            hits = rag.retrieve(question, top_k=3)
            if hits:
                for h in hits:
                    with st.expander(f"📄 {h['source']} — p.{h['page']}  ·  score {h['score']:.2f}"):
                        st.write(h["text"][:500] + ("…" if len(h["text"]) > 500 else ""))
            else:
                st.info("No PDFs indexed. Evaluation will use LLM general knowledge.")
        else:
            st.markdown('<div class="fb-box">Enter a question above to preview retrieved context.</div>', unsafe_allow_html=True)

    if evaluate:
        if not question.strip() or not answer.strip():
            st.error("Please enter both a question and a student answer.")
        else:
            with st.spinner("Retrieving context and evaluating…"):
                context   = rag.get_context(question)
                evaluator = Evaluator(api_key)
                t0        = time.time()
                result    = evaluator.evaluate(question, answer, context)
                elapsed   = round(time.time() - t0, 1)

            if "error" in result:
                st.error(f"Evaluation failed: {result['error']}")
            else:
                scale = max_marks / 10
                result["marks_awarded"] = round(result["marks_awarded"] * scale)
                result["max_marks"]     = max_marks
                grade = result["grade"]
                st.session_state.history.append({"question": question, "grade": grade, "marks": result["marks_awarded"]})

                st.markdown("---")
                st.markdown('<div class="section-label">Evaluation Results</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metrics-row">
                    <div class="metric-box"><div class="label">Score</div><div class="value" style="color:{grade_color(grade)}">{result['marks_awarded']}<span style="font-size:1rem;color:#8b949e">/{max_marks}</span></div></div>
                    <div class="metric-box"><div class="label">Percentage</div><div class="value">{result['percentage']}%</div></div>
                    <div class="metric-box"><div class="label">Grade</div><div class="value" style="color:{grade_color(grade)}">{grade}</div></div>
                    <div class="metric-box"><div class="label">Eval Time</div><div class="value" style="font-size:1.4rem">{elapsed}s</div></div>
                </div>
                """, unsafe_allow_html=True)

                r1, r2 = st.columns([1, 1], gap="large")
                with r1:
                    st.plotly_chart(score_gauge(result["marks_awarded"], max_marks), use_container_width=True)
                    st.markdown('<div class="section-label">Concepts Covered ✅</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chips-wrap">{chips_html(result["concepts_covered"], "chip-green")}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-label" style="margin-top:1rem">Concepts Missing ❌</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chips-wrap">{chips_html(result["concepts_missing"], "chip-red")}</div>', unsafe_allow_html=True)

                with r2:
                    st.markdown('<div class="section-label">Strengths</div>', unsafe_allow_html=True)
                    items = "".join(f'<div class="sw-item"><div class="sw-dot dot-green"></div>{s}</div>' for s in result.get("strengths", []))
                    st.markdown(f'<div class="card" style="padding:0.8rem 1rem">{items or "<span style=color:#8b949e>None</span>"}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-label" style="margin-top:1rem">Areas to Improve</div>', unsafe_allow_html=True)
                    items2 = "".join(f'<div class="sw-item"><div class="sw-dot dot-red"></div>{w}</div>' for w in result.get("weaknesses", []))
                    st.markdown(f'<div class="card" style="padding:0.8rem 1rem">{items2 or "<span style=color:#8b949e>None</span>"}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-label" style="margin-top:1rem">Detailed Feedback</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="fb-box">{result["detailed_feedback"]}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-label" style="margin-top:1.5rem">💡 Model Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="fb-box green">{result["improved_answer"]}</div>', unsafe_allow_html=True)
                if context:
                    with st.expander("📚 View Retrieved Reference Context"):
                        st.code(context, language=None)

with tab2:
    st.markdown('<div class="section-label">Batch Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="fb-box">Upload a CSV with columns: <code>student_name</code>, <code>question</code>, <code>student_answer</code></div>', unsafe_allow_html=True)
    sample = "student_name,question,student_answer\nAlice,What is RAG?,RAG stands for Retrieval Augmented Generation.\nBob,What is FAISS?,FAISS is a library for fast similarity search."
    st.download_button("📥 Download Sample CSV", sample, "sample_batch.csv", "text/csv")
    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch")
    if batch_file and api_key:
        df = pd.read_csv(batch_file)
        st.dataframe(df.head(), use_container_width=True)
        if st.button("🚀 Run Batch Evaluation", type="primary"):
            evaluator = Evaluator(api_key)
            results, prog, status = [], st.progress(0), st.empty()
            for i, row in df.iterrows():
                status.text(f"Evaluating {i+1}/{len(df)}…")
                q, ans = str(row.get("question","")), str(row.get("student_answer",""))
                res = evaluator.evaluate(q, ans, rag.get_context(q))
                results.append({"Student": row.get("student_name", f"Student {i+1}"), "Question": q[:50]+"…", "Marks": res.get("marks_awarded",0), "Grade": res.get("grade","N/A"), "Percentage": res.get("percentage",0), "Feedback": res.get("detailed_feedback","")[:100]+"…"})
                prog.progress((i+1)/len(df))
            status.empty()
            rdf = pd.DataFrame(results)
            st.dataframe(rdf, use_container_width=True)
            st.download_button("📥 Download Results", rdf.to_csv(index=False), "results.csv", "text/csv")
            gc = rdf["Grade"].value_counts()
            fig = go.Figure(go.Bar(x=gc.index.tolist(), y=gc.values.tolist(), marker_color=[grade_color(g) for g in gc.index], marker_line_width=0))
            fig.update_layout(title="Grade Distribution", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font={"color":"#e8e8f0"}, height=300, xaxis={"gridcolor":"#21262d"}, yaxis={"gridcolor":"#21262d"})
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-label">System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""<div class="fb-box" style="font-family:monospace;font-size:0.82rem;line-height:1.8;white-space:pre">
┌─────────────────────────────────────────────────────────┐
│              OFFLINE — Knowledge Base Setup             │
│  Syllabus PDFs → Text Extraction → Chunking (500 chars) │
│         Sentence Transformer Embeddings                 │
│              (all-MiniLM-L6-v2, 384-dim)                │
│           FAISS IndexFlatIP (cosine similarity)         │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│              ONLINE — Evaluation Pipeline               │
│  Question + Student Answer                              │
│  → Embed → FAISS Search → Top-5 Chunks                  │
│  → Prompt = System + Question + Answer + Context        │
│  → Groq LLaMA-3.3 70B Versatile                         │
│  → JSON: Marks · Grade · Concepts · Feedback            │
└─────────────────────────────────────────────────────────┘</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem">Technology Stack</div>', unsafe_allow_html=True)
    for name, version, desc in [
        ("Sentence Transformers","all-MiniLM-L6-v2","Converts text to 384-dim semantic vectors"),
        ("FAISS","IndexFlatIP","Fast cosine similarity search over embeddings"),
        ("Groq API","LLaMA-3.3 70B","LLM for concept extraction, scoring, feedback"),
        ("PyMuPDF","fitz","PDF text extraction page by page"),
        ("Streamlit","1.35+","Interactive web UI framework"),
        ("Plotly","5.x","Score gauge and grade distribution charts"),
    ]:
        st.markdown(f"<div class='sw-item'><div class='sw-dot' style='background:#58a6ff;margin-top:7px'></div><div><span style='color:#e8e8f0;font-weight:500'>{name}</span> <span style='color:#58a6ff;font-size:0.78rem'>({version})</span> <span style='color:#8b949e;font-size:0.85rem'>— {desc}</span></div></div>", unsafe_allow_html=True)
