import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi


# ==========================
# Load FAISS
# ==========================

print("Loading FAISS...")

index = faiss.read_index(
    "embeddings/faiss.index"
)


with open(
    "embeddings/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)


print(
    "Loaded chunks:",
    len(chunks)
)


# ==========================
# Load Embedding Model
# ==========================

print("Loading embedding model...")


embed_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# ==========================
# BM25 Setup
# ==========================

print("Creating BM25 index...")


tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]


bm25 = BM25Okapi(
    tokenized_chunks
)



# ==========================
# Query
# ==========================

query = input(
    "\nEnter your question: "
)


# ==========================
# BM25 Retrieval
# ==========================

bm25_scores = bm25.get_scores(
    query.lower().split()
)


bm25_top = np.argsort(
    bm25_scores
)[::-1][:5]



# ==========================
# FAISS Retrieval
# ==========================


query_embedding = embed_model.encode(
    [query],
    normalize_embeddings=True
)


faiss_scores, faiss_indices = index.search(
    query_embedding,
    5
)



# ==========================
# Score Fusion
# ==========================

results = {}


# BM25 weight

for idx in bm25_top:

    results[idx] = (
        results.get(idx,0)
        +
        0.5
    )


# FAISS weight

for rank,idx in enumerate(faiss_indices[0]):

    results[idx] = (
        results.get(idx,0)
        +
        (1/(rank+1))
    )



# Sort

final_results = sorted(
    results.items(),
    key=lambda x:x[1],
    reverse=True
)



# ==========================
# Output
# ==========================

print("\n========== HYBRID RESULTS ==========")


for i,(idx,score) in enumerate(
    final_results[:5]
):

    print(
        "\nResult",
        i+1,
        "Score:",
        score
    )

    print(
        chunks[idx][:700]
    )