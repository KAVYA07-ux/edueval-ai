"""
RAG Engine — PDF ingestion, chunking, embedding, and Pinecone vector storage.
Replaces the local FAISS index with a persistent cloud vector DB.
"""

import os
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# ── Constants ────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim embeddings
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
TOP_K            = 5
INDEX_NAME       = "edueval-rag"         # Pinecone index name
DIMENSION        = 384                   # must match embedding model


class RAGEngine:
    """Manages the full Retrieve-Augment pipeline using Pinecone."""

    def __init__(self, pinecone_api_key: str):
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self._pc = Pinecone(api_key=pinecone_api_key)
        self._index = self._get_or_create_index()

    # ── Index lifecycle ──────────────────────────────────────────────────────

    def _get_or_create_index(self):
        """Return an existing Pinecone index or create one."""
        existing = [i.name for i in self._pc.list_indexes()]
        if INDEX_NAME not in existing:
            self._pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        return self._pc.Index(INDEX_NAME)

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
        """Chunk, embed, and upsert PDF content into Pinecone. Returns chunks added."""
        vectors = []

        for path in pdf_paths:
            pages = self.extract_text_from_pdf(path)
            for p in pages:
                splits = self.splitter.split_text(p["text"])
                for i, chunk in enumerate(splits):
                    embedding = self._embed([chunk])[0].tolist()
                    chunk_id = f"{p['source']}_p{p['page']}_c{i}"
                    vectors.append({
                        "id":     chunk_id,
                        "values": embedding,
                        "metadata": {
                            "text":   chunk,
                            "source": p["source"],
                            "page":   p["page"],
                        },
                    })

        if not vectors:
            return 0

        # Upsert in batches of 100 (Pinecone limit per request)
        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            self._index.upsert(vectors=vectors[start : start + batch_size])

        return len(vectors)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Return top-k most relevant chunks for a query."""
        stats = self._index.describe_index_stats()
        if stats.total_vector_count == 0:
            return []

        q_emb = self._embed([query])[0].tolist()
        results = self._index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True,
        )

        hits = []
        for match in results.matches:
            meta = match.metadata or {}
            hits.append({
                "text":   meta.get("text", ""),
                "source": meta.get("source", "unknown"),
                "page":   meta.get("page", 0),
                "score":  round(match.score, 3),
            })
        return hits

    def get_context(self, query: str, top_k: int = TOP_K) -> str:
        """Return a single concatenated context string for LLM prompt injection."""
        hits = self.retrieve(query, top_k)
        if not hits:
            return ""
        parts = []
        for i, h in enumerate(hits, 1):
            parts.append(
                f"[Reference {i} — {h['source']}, p.{h['page']} | relevance {h['score']:.2f}]\n{h['text']}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Stats / management ───────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        try:
            return self._index.describe_index_stats().total_vector_count
        except Exception:
            return 0

    def clear_index(self):
        """Delete all vectors from the Pinecone index."""
        self._index.delete(delete_all=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
