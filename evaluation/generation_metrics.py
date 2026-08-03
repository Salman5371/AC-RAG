# =====================================================
# AC-RAG Generation Evaluation
# Keyword Coverage + Semantic Similarity
# =====================================================


import os
import sys
import numpy as np


# =====================================================
# Add Project Root Path
# =====================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)



from sentence_transformers import (
    SentenceTransformer,
    util
)


from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.adaptive_retriever import AdaptiveRetriever


from generation.qwen_generator import QwenGenerator




print("\nLoading AC-RAG...")



# =====================================================
# Load Retrieval System
# =====================================================


retriever = HybridRetriever()


reranker = Reranker()


adaptive = AdaptiveRetriever(
    retriever,
    reranker
)



# =====================================================
# Load Qwen Generator
# =====================================================


generator = QwenGenerator()



# =====================================================
# Semantic Evaluation Model
# =====================================================


print("\nLoading evaluation model...")


semantic_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)




# =====================================================
# Test Questions
# =====================================================


questions = [


{
"question":
"What is Retrieval Augmented Generation?",


"keywords":[
"retrieval",
"external knowledge",
"large language models",
"hallucination"
],


"reference":
"""
Retrieval Augmented Generation (RAG) improves
Large Language Models by retrieving relevant
information from external knowledge sources.
It reduces hallucination by grounding generated
answers with retrieved documents.
"""
},



{
"question":
"How does RAG reduce hallucination?",


"keywords":[
"external knowledge",
"retrieved documents",
"factual"
],


"reference":
"""
RAG reduces hallucination by retrieving relevant
documents from external knowledge bases and
using those documents as context to generate
more accurate factual responses.
"""
},



{
"question":
"What are the methods used to improve retrieval quality in RAG systems?",


"keywords":[
"query rewriting",
"metadata",
"reranking",
"index"
],


"reference":
"""
Retrieval quality can be improved using query
optimization, query rewriting, metadata,
better indexing strategies, hybrid retrieval,
and reranking.
"""
},



{
"question":
"Explain Advanced RAG.",


"keywords":[
"pre-retrieval",
"post-retrieval",
"indexing"
],


"reference":
"""
Advanced RAG improves traditional RAG using
pre-retrieval and post-retrieval strategies,
better indexing techniques and reranking.
"""
},



{
"question":
"What are the limitations of Large Language Models?",


"keywords":[
"hallucination",
"outdated knowledge",
"accuracy"
],


"reference":
"""
Large Language Models suffer from hallucination,
outdated knowledge and factual accuracy problems,
especially for knowledge-intensive tasks.
"""
}


]




keyword_scores = []

semantic_scores = []



# =====================================================
# Evaluation Loop
# =====================================================


for item in questions:


    question = item["question"]


    print("\n================================")
    print("Question:")
    print(question)



    # -----------------------------
    # Retrieve Context
    # -----------------------------


    docs = adaptive.retrieve(
        question
    )



    context_parts = []



    for doc in docs:


        if isinstance(doc, dict):

            context_parts.append(
                doc.get(
                    "text",
                    ""
                )
            )


        else:

            context_parts.append(
                str(doc)
            )



    context = "\n\n".join(
        context_parts
    )



    # -----------------------------
    # Generate Answer
    # -----------------------------


    answer = generator.generate(
        question,
        context
    )



    print("\nAnswer:")
    print(
        answer[:600]
    )



    # -----------------------------
    # Keyword Coverage
    # -----------------------------


    matched = 0


    for key in item["keywords"]:


        if key.lower() in answer.lower():

            matched += 1



    keyword_score = (
        matched /
        len(item["keywords"])
    )


    keyword_scores.append(
        keyword_score
    )



    print(
        "\nKeyword Coverage:",
        matched,
        "/",
        len(item["keywords"])
    )



    # -----------------------------
    # Semantic Similarity
    # -----------------------------


    answer_embedding = semantic_model.encode(
        answer,
        convert_to_tensor=True
    )


    reference_embedding = semantic_model.encode(
        item["reference"],
        convert_to_tensor=True
    )



    similarity = util.cos_sim(
        answer_embedding,
        reference_embedding
    ).item()



    semantic_scores.append(
        similarity
    )



    print(
        "Semantic Similarity:",
        round(similarity,3)
    )





# =====================================================
# Final Score
# =====================================================


avg_keyword = np.mean(
    keyword_scores
)


avg_semantic = np.mean(
    semantic_scores
)



overall = (

    avg_keyword * 0.3

    +

    avg_semantic * 0.7

)



print(
    "\n========== GENERATION RESULTS =========="
)


print(
    "Questions:",
    len(questions)
)


print(
    "Average Keyword Coverage:",
    round(avg_keyword,3)
)


print(
    "Average Semantic Similarity:",
    round(avg_semantic,3)
)


print(
    "Overall Generation Score:",
    round(overall,3)
)