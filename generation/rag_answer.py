# =====================================================
# AC-RAG Final Answer Pipeline
# Adaptive Retrieval + Qwen Generation + Sources
# =====================================================


import sys
import os



# ==========================
# Add Project Root
# ==========================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.append(
    PROJECT_ROOT
)



# ==========================
# Imports
# ==========================

from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.adaptive_retriever import AdaptiveRetriever

from generation.qwen_generator import QwenGenerator




print(
    "\n========== AC-RAG SYSTEM ==========\n"
)



# ==========================
# Load Retrieval System
# ==========================

print(
    "Loading Hybrid Retriever..."
)


retriever = HybridRetriever()



print(
    "Loading Reranker..."
)


reranker = Reranker()



adaptive_retriever = AdaptiveRetriever(
    retriever,
    reranker
)



# ==========================
# Load Generator
# ==========================

print(
    "Loading Qwen Generator..."
)


generator = QwenGenerator()




# ==========================
# Question
# ==========================

query = input(
    "\nEnter question: "
)




# ==========================
# Retrieve Context
# ==========================

documents = adaptive_retriever.retrieve(
    query
)



print(
    "\nSelected Evidence:",
    len(documents)
)




# ==========================
# Build Context
# ==========================

context = ""


for doc in documents:


    context += doc.get(
        "text",
        ""
    )


    context += "\n\n"




# ==========================
# Generate Answer
# ==========================

answer = generator.generate(
    query,
    context
)




# ==========================
# Final Answer
# ==========================

print(
    "\n========== FINAL ANSWER ==========\n"
)


print(
    answer
)




# ==========================
# Sources
# ==========================

print(
    "\n========== SOURCES ==========\n"
)



for i,doc in enumerate(documents):


    print(
        "Source:",
        i+1
    )


    print(
        "Chunk ID:",
        doc.get(
            "chunk_id"
        )
    )


    print(
        "FAISS Score:",
        doc.get(
            "faiss_score"
        )
    )


    print(
        "Hybrid Score:",
        doc.get(
            "hybrid_score"
        )
    )


    print(
        "Reranker Score:",
        doc.get(
            "reranker_score"
        )
    )


    print(
        "---------------------------"
    )