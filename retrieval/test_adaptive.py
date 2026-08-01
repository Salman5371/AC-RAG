from retriever import HybridRetriever
from reranker import Reranker
from adaptive_retriever import AdaptiveRetriever


print("Loading system...")


retriever = HybridRetriever()

reranker = Reranker()


adaptive = AdaptiveRetriever(
    retriever,
    reranker
)


query = input(
    "Enter question: "
)


results = adaptive.retrieve(
    query
)


print("\n========== FINAL RESULTS ==========")


for i,item in enumerate(results):

    print(
        "\nResult:",
        i+1
    )


    if isinstance(item, tuple):

        text = item[0]
        score = item[1]

        print(
            "Score:",
            float(score)
        )

        print(
            "\nContext:"
        )

        print(
            text[:700]
        )


    elif isinstance(item, dict):

        print(
            item["score"]
        )

        print(
            item["text"][:700]
        )


    else:

        print(
            item[:700]
        )