# =====================================================
# AC-RAG Adaptive Retrieval Test
# =====================================================


from retriever import HybridRetriever

from adaptive_retriever import AdaptiveRetriever




print(
    "\n========== AC-RAG ADAPTIVE TEST =========="
)



retriever = HybridRetriever()



adaptive = AdaptiveRetriever(

    retriever

)




questions = [


    "What is Retrieval Augmented Generation?",


    "How does RAG reduce hallucination?",


    "What are the methods used to improve retrieval quality in RAG systems?",


    "Explain the role of quantum computing in improving RAG retrieval systems."

]




for i, question in enumerate(questions):


    print(
        "\n\n=============================="
    )


    print(
        "QUESTION:",
        i+1
    )


    print(
        question
    )


    print(
        "=============================="
    )



    results = adaptive.retrieve(

        question

    )



    print(
        "\nFINAL RESULTS:",
        len(results)
    )



    for idx, doc in enumerate(results):


        print(
            "\nResult:",
            idx+1
        )


        print(
            "Chunk ID:",
            doc.get(
                "chunk_id"
            )
        )


        print(
            "Reranker Score:",
            doc.get(
                "reranker_score"
            )
        )


        print(
            "Context:"
        )


        print(
            doc["text"][:300]
        )



print(
    "\n========== TEST COMPLETE =========="
)