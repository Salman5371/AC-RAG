import faiss
import pickle

from sentence_transformers import SentenceTransformer


# ======================
# Load FAISS Database
# ======================

print("Loading FAISS index...")

index = faiss.read_index(
    "embeddings/faiss.index"
)


with open(
    "embeddings/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)


print(
    "Chunks loaded:",
    len(chunks)
)


# ======================
# Load Embedding Model
# ======================

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# ======================
# User Query
# ======================

query = input(
    "\nEnter your question: "
)


query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)


# ======================
# Search
# ======================

k = 5


scores, indices = index.search(
    query_embedding,
    k
)


print("\n===== Retrieved Context =====")


for i, idx in enumerate(indices[0]):

    print("\n--- Result", i+1, "---")

    print(
        "Score:",
        scores[0][i]
    )

    print(
        chunks[idx]
    )