from sentence_transformers import CrossEncoder

import pickle
import faiss

from sentence_transformers import SentenceTransformer


# ==========================
# Load chunks
# ==========================

print("Loading chunks...")

with open(
    "embeddings/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)



# ==========================
# Load FAISS
# ==========================

index = faiss.read_index(
    "embeddings/faiss.index"
)



# ==========================
# Embedding model
# ==========================

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)



# ==========================
# Reranker Model
# ==========================

print("Loading reranker...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)



# ==========================
# Query
# ==========================

query = input(
    "\nEnter question: "
)



# ==========================
# Retrieve candidates
# ==========================

query_embedding = embedding_model.encode(
    [query],
    normalize_embeddings=True
)


scores, indices = index.search(
    query_embedding,
    20
)


candidates = []


for idx in indices[0]:

    candidates.append(
        chunks[idx]
    )



print(
    "\nCandidates:",
    len(candidates)
)



# ==========================
# Reranking
# ==========================

pairs = []


for chunk in candidates:

    pairs.append(
        [
            query,
            chunk
        ]
    )


rerank_scores = reranker.predict(
    pairs
)



results = sorted(
    zip(
        candidates,
        rerank_scores
    ),
    key=lambda x:x[1],
    reverse=True
)



# ==========================
# Output
# ==========================

print(
    "\n========== RERANKED RESULTS =========="
)


for i,(text,score) in enumerate(results[:5]):

    print(
        "\nResult",
        i+1,
        "Score:",
        score
    )

    print(
        text[:700]
    )