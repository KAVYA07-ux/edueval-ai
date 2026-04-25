import os
import faiss
import pickle
import pdfplumber
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Folder containing reference PDFs / text
DATA_PATH = "data/"
INDEX_PATH = "vector_db/faiss_index.pkl"


# -------------------------------
# Step 1: Extract text from PDFs
# -------------------------------
def load_documents(data_path):
    documents = []

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(data_path, file)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                documents.append(text)

        elif file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                documents.append(f.read())

    return documents


# -------------------------------
# Step 2: Chunk text
# -------------------------------
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -------------------------------
# Step 3: Create FAISS Index
# -------------------------------
def create_vector_store():
    documents = load_documents(DATA_PATH)
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)

    # Generate embeddings
    embeddings = model.encode(all_chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index and chunks
    with open(INDEX_PATH, "wb") as f:
        pickle.dump((index, all_chunks), f)

    print("Vector DB created and saved!")


# -------------------------------
# Step 4: Load Vector Store
# -------------------------------
def load_vector_store():
    with open(INDEX_PATH, "rb") as f:
        index, chunks = pickle.load(f)
    return index, chunks


# -------------------------------
# Step 5: Retrieve Relevant Context
# -------------------------------
def retrieve_context(query, top_k=3):
    index, chunks = load_vector_store()

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = [chunks[i] for i in indices[0]]
    return "\n\n".join(results)


if __name__ == "__main__":
    create_vector_store()
