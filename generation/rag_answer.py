# =====================================================
# AC-RAG Final Pipeline
# Adaptive Retrieval + Reranking + Qwen Generation
# =====================================================


import sys
import os


# ==========================
# Add Project Root Path
# ==========================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)



# ==========================
# Imports
# ==========================

from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.adaptive_retriever import AdaptiveRetriever

from generation.qwen_generator import QwenGenerator



print("\n========== AC-RAG SYSTEM ==========\n")



# ==========================
# Load Retriever
# ==========================

print("Loading Hybrid Retriever...")

retriever = HybridRetriever()



# ==========================
# Load Reranker
# ==========================

print("Loading Reranker...")

reranker = Reranker()



# ==========================
# Adaptive Retrieval
# ==========================

adaptive_retriever = AdaptiveRetriever(
    retriever,
    reranker
)



# ==========================
# Load Generator
# ==========================

print("Loading Qwen Generator...")

generator = QwenGenerator()



# ==========================
# User Query
# ==========================

query = input(
    "\nEnter question: "
)



# ==========================
# Retrieve Documents
# ==========================

documents = adaptive_retriever.retrieve(
    query
)



print(
    "\nRetrieved documents:",
    len(documents)
)



# ==========================
# Context Construction
# ==========================

context = ""


for doc in documents:


    if isinstance(doc, dict):

        context += doc.get(
            "text",
            ""
        )


    elif isinstance(doc, tuple):

        context += str(
            doc[0]
        )


    else:

        context += str(doc)



    context += "\n\n"



# ==========================
# Generate Answer
# ==========================

answer = generator.generate(
    query,
    context
)



# ==========================
# Final Output
# ==========================

print(
    "\n========== FINAL ANSWER ==========\n"
)


print(answer)