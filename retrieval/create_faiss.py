import os
import pickle
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -------------------------
# Load PDF
# -------------------------

pdf_path = "documents/rag_paper.pdf"


reader = PdfReader(pdf_path)


text = ""

for page in reader.pages:
    text += page.extract_text()


print("PDF loaded")


# -------------------------
# Chunking
# -------------------------

chunk_size = 500

chunks = []


for i in range(0, len(text), chunk_size):

    chunks.append(
        text[i:i+chunk_size]
    )


print("Chunks:", len(chunks))


# -------------------------
# Embedding Model
# -------------------------

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)


# -------------------------
# FAISS Index
# -------------------------

dimension = embeddings.shape[1]


index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embeddings
)


print(
    "FAISS index created"
)


# Save

os.makedirs(
    "embeddings",
    exist_ok=True
)


faiss.write_index(
    index,
    "embeddings/faiss.index"
)


with open(
    "embeddings/chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )


print("Saved successfully")