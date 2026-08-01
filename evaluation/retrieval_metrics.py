# =====================================================
# AC-RAG Retrieval Evaluation
# Precision@K + Recall@K + Hit Rate
# =====================================================


import json
import sys
import os



PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(
    PROJECT_ROOT
)



from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.adaptive_retriever import AdaptiveRetriever




# ==========================
# Load Test Questions
# ==========================


with open(
    "evaluation/test_questions.json",
    "r",
    encoding="utf-8"
) as f:

    test_questions = json.load(f)




# ==========================
# Load AC-RAG
# ==========================


print(
    "\nLoading AC-RAG..."
)


retriever = HybridRetriever()

reranker = Reranker()


adaptive = AdaptiveRetriever(
    retriever,
    reranker
)




# ==========================
# Evaluation
# ==========================


total_questions = len(
    test_questions
)


hits = 0

precision_scores = []



for item in test_questions:


    question = item["question"]


    keywords = [
        k.lower()
        for k in item["expected_keywords"]
    ]



    print(
        "\nQuestion:",
        question
    )



    results = adaptive.retrieve(
        question
    )



    retrieved_text = ""


    for doc in results:


        retrieved_text += (
            doc.get(
                "text",
                ""
            )
            .lower()
        )



    matched = 0



    for keyword in keywords:


        if keyword in retrieved_text:

            matched += 1



    precision = (
        matched /
        len(keywords)
    )



    precision_scores.append(
        precision
    )



    if matched > 0:

        hits += 1



    print(
        "Keyword Match:",
        matched,
        "/",
        len(keywords)
    )



# ==========================
# Final Metrics
# ==========================


precision_at_k = (
    sum(precision_scores)
    /
    total_questions
)


hit_rate = (
    hits /
    total_questions
)



print(
    "\n========== RESULTS =========="
)


print(
    "Questions:",
    total_questions
)


print(
    "Precision@K:",
    round(
        precision_at_k,
        3
    )
)


print(
    "Hit Rate:",
    round(
        hit_rate,
        3
    )
)