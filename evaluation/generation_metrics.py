# =====================================================
# AC-RAG Generation Evaluation
# Answer Relevance + Context Usage
# =====================================================


import json
import os
import sys



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

from generation.qwen_generator import QwenGenerator




# ==========================
# Load Questions
# ==========================


with open(
    "evaluation/test_questions.json",
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)




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


generator = QwenGenerator()




# ==========================
# Evaluation
# ==========================


total_score = 0



for item in questions:


    query = item["question"]


    keywords = [

        k.lower()

        for k in item["expected_keywords"]

    ]



    print(
        "\nQuestion:",
        query
    )



    # Retrieve

    docs = adaptive.retrieve(
        query
    )



    context = ""



    for doc in docs:


        context += doc.get(
            "text",
            ""
        )



        context += "\n"




    # Generate answer


    answer = generator.generate(
        query,
        context
    )



    answer_lower = answer.lower()



    matched = 0



    for keyword in keywords:


        if keyword in answer_lower:

            matched += 1




    score = (
        matched /
        len(keywords)
    )



    total_score += score



    print(
        "\nAnswer:"
    )


    print(
        answer[:500]
    )


    print(
        "\nKeyword Coverage:",
        matched,
        "/",
        len(keywords)
    )




# ==========================
# Final Result
# ==========================


average_score = (

    total_score /
    len(questions)

)



print(
    "\n========== GENERATION RESULTS =========="
)



print(
    "Questions:",
    len(questions)
)


print(
    "Answer Coverage Score:",
    round(
        average_score,
        3
    )
)