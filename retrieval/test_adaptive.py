# =====================================================
# AC-RAG Adaptive Retrieval Test
# Hybrid Retrieval + Reranking + Metadata Output
# =====================================================


from retriever import HybridRetriever
from reranker import Reranker
from adaptive_retriever import AdaptiveRetriever



print("\n========== AC-RAG ADAPTIVE TEST ==========\n")



# ==========================
# Load Retriever
# ==========================

retriever = HybridRetriever()



# ==========================
# Load Reranker
# ==========================

reranker = Reranker()



# ==========================
# Adaptive Retriever
# ==========================

adaptive = AdaptiveRetriever(
    retriever,
    reranker
)



# ==========================
# User Query
# ==========================

query = input(
    "\nEnter question: "
)



# ==========================
# Retrieve + Rerank
# ==========================

results = adaptive.retrieve(
    query
)



# ==========================
# Display Results
# ==========================

print(
    "\n========== FINAL RESULTS =========="
)



for i, item in enumerate(results):


    print(
        "\nResult:",
        i + 1
    )


    print(
        "--------------------------------"
    )


    # Metadata output

    print(
        "Chunk ID:",
        item.get(
            "chunk_id",
            "N/A"
        )
    )


    print(
        "BM25 Score:",
        item.get(
            "bm25_score",
            0
        )
    )


    print(
        "FAISS Score:",
        item.get(
            "faiss_score",
            0
        )
    )


    print(
        "Hybrid Score:",
        item.get(
            "hybrid_score",
            0
        )
    )


    print(
        "Reranker Score:",
        item.get(
            "reranker_score",
            0
        )
    )


    print(
        "\nContext:"
    )


    print(
        item.get(
            "text",
            ""
        )[:700]
    )



print(
    "\n========== TEST COMPLETE =========="
)