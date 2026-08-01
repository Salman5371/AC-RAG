# =====================================================
# AC-RAG Adaptive Retriever
# Adaptive Query Analysis + Hybrid Retrieval + Reranking
# Final Stable Version
# =====================================================


import os
import sys


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.append(CURRENT_DIR)


from reranker import Reranker



class AdaptiveRetriever:


    def __init__(
        self,
        retriever,
        reranker=None
    ):


        print(
            "Loading Adaptive Retrieval System..."
        )


        self.retriever = retriever


        if reranker:

            self.reranker = reranker

        else:

            self.reranker = Reranker()



    # =====================================================
    # Query Complexity Detection
    # =====================================================


    def analyze_query(
        self,
        query
    ):


        words = query.split()


        complex_words = [
            "methods",
            "techniques",
            "explain",
            "compare",
            "advantages",
            "limitations",
            "improve",
            "framework",
            "architecture",
            "process",
            "everything",
            "detailed"
        ]


        score = 0


        for word in complex_words:

            if word.lower() in query.lower():

                score += 1



        if len(words) > 10:

            score += 1



        if score >= 2:

            return "Complex"


        else:

            return "Simple"




    # =====================================================
    # Retrieval Quality Check
    # =====================================================


    def check_quality(
        self,
        docs
    ):


        if not docs:

            return False



        total = 0


        for doc in docs:


            if isinstance(doc,dict):

                text = doc.get(
                    "text",
                    ""
                )


            else:

                text = str(doc)



            total += len(text)



        avg = total / len(docs)


        print(
            "Average document length:",
            avg
        )


        return avg > 200




    # =====================================================
    # Query Reformulation
    # =====================================================


    def reformulate_query(
        self,
        query
    ):


        new_query = (
            "Provide detailed academic information about: "
            +
            query
        )


        print(
            "Reformulated Query:",
            new_query
        )


        return new_query




    # =====================================================
    # Main Adaptive Retrieval Pipeline
    # =====================================================


    def retrieve(
        self,
        query
    ):


        print(
            "\n[Adaptive Retrieval]"
        )


        print(
            "Query:",
            query
        )



        # -------------------------
        # Analyze Query
        # -------------------------


        query_type = self.analyze_query(
            query
        )


        print(
            "Query Type:",
            query_type
        )



        # -------------------------
        # Dynamic Retrieval Size
        # -------------------------


        if query_type == "Complex":

            top_k = 40

        else:

            top_k = 20



        # -------------------------
        # First Retrieval
        # -------------------------


        docs = self.retriever.search(
            query,
            top_k=top_k
        )


        print(
            "Retrieved documents:",
            len(docs)
        )



        # -------------------------
        # Quality Check
        # -------------------------


        quality = self.check_quality(
            docs
        )


        if not quality:


            print(
                "Low quality detected, reformulating query..."
            )


            query = self.reformulate_query(
                query
            )


            docs = self.retriever.search(
                query,
                top_k=top_k
            )



        # -------------------------
        # Reranking
        # -------------------------


        print(
            "\nReranking documents..."
        )


        ranked_docs = self.reranker.rerank(
            query,
            docs
        )



        # -------------------------
        # Stable Selection
        # -------------------------


        if len(ranked_docs) >= 5:


            final_docs = ranked_docs[:5]


        else:

            final_docs = ranked_docs



        print(
            "Final selected documents:",
            len(final_docs)
        )


        return final_docs