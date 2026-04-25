"""
RAG Engine — handles PDF ingestion, chunking, embedding, FAISS indexing,
and semantic retrieval for the Answer Evaluation System.
"""

import os
import pickle
import numpy as np
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Constants ────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 80 MB, runs locally, no API needed
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
TOP_K            = 5
INDEX_PATH       = "vector_store/faiss.index"
META_PATH        = "vector_store/metadata.pkl"


class RAGEngine:
    """Manages the full Retrieve-Augment pipeline."""

    def __init__(self):
        self.model      = SentenceTransformer(EMBED_MODEL_NAME)
        self.index      = None          # FAISS index
        self.chunks     = []            # raw text chunks
        self.sources    = []            # (filename, page_num) per chunk
        self.splitter   = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    # ── PDF ingestion ────────────────────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_path: str) -> list[dict]:
        """Return list of {text, page, source} dicts from a PDF file."""
        pages = []
        doc   = fitz.open(pdf_path)
        fname = os.path.basename(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"text": text, "page": page_num, "source": fname})
        doc.close()
        return pages

    def add_documents(self, pdf_paths: list[str]) -> int:
        """Chunk, embed, and index a list of PDF files. Returns total chunks added."""
        new_chunks  = []
        new_sources = []

        for path in pdf_paths:
            pages = self.extract_text_from_pdf(path)
            for p in pages:
                splits = self.splitter.split_text(p["text"])
                for s in splits:
                    new_chunks.append(s)
                    new_sources.append((p["source"], p["page"]))

        if not new_chunks:
            return 0

        embeddings = self._embed(new_chunks)                 # (N, D)

        if self.index is None:
            dim         = embeddings.shape[1]
            self.index  = faiss.IndexFlatIP(dim)             # Inner product ≈ cosine (after normalisation)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks  += new_chunks
        self.sources += new_sources

        self.save_index()
        return len(new_chunks)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Return top-k most relevant chunks for a query."""
        if self.index is None or self.index.ntotal == 0:
            return []

        q_emb = self._embed([query])
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text":   self.chunks[idx],
                "source": self.sources[idx][0],
                "page":   self.sources[idx][1],
                "score":  float(score),
            })
        return results

    def get_context(self, query: str, top_k: int = TOP_K) -> str:
        """Return a single concatenated context string for injection into an LLM prompt."""
        hits = self.retrieve(query, top_k)
        if not hits:
            return ""
        parts = []
        for i, h in enumerate(hits, 1):
            parts.append(
                f"[Reference {i} — {h['source']}, p.{h['page']} | relevance {h['score']:.2f}]\n{h['text']}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_index(self):
        os.makedirs("vector_store", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "sources": self.sources}, f)

    def load_index(self) -> bool:
        """Load persisted index. Returns True on success."""
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            return False
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta          = pickle.load(f)
            self.chunks   = meta["chunks"]
            self.sources  = meta["sources"]
        return True

    def clear_index(self):
        self.index   = None
        self.chunks  = []
        self.sources = []
        for p in [INDEX_PATH, META_PATH]:
            if os.path.exists(p):
                os.remove(p)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
