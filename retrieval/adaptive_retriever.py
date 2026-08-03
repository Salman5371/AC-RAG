# =====================================================
# AC-RAG Adaptive Retriever
# Adaptive Query Analysis + Hybrid Retrieval + Reranking
# Balanced Generation Optimized Version
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

            if word in query.lower():

                score += 1



        if len(query.split()) > 10:

            score += 1



        if score >= 2:

            return "Complex"


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



        total_length = 0



        for doc in docs:


            if isinstance(doc, dict):

                text = doc.get(
                    "text",
                    ""
                )


            else:

                text = str(doc)



            total_length += len(text)



        avg_length = (
            total_length /
            len(docs)
        )



        print(
            "Average document length:",
            avg_length
        )


        return avg_length > 200




    # =====================================================
    # Query Reformulation
    # =====================================================


    def reformulate_query(
        self,
        query
    ):


        new_query = (

            "Provide academic explanation with important concepts about: "
            +
            query

        )


        print(
            "Reformulated Query:",
            new_query
        )


        return new_query




    # =====================================================
    # Adaptive Retrieval Pipeline
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



        # Query analysis

        query_type = self.analyze_query(
            query
        )


        print(
            "Query Type:",
            query_type
        )



        # Dynamic retrieval size

        if query_type == "Complex":

            top_k = 70


        else:

            top_k = 40




        # Initial retrieval

        docs = self.retriever.search(
            query,
            top_k=top_k
        )



        print(
            "Retrieved documents:",
            len(docs)
        )



        # Quality checking

        if not self.check_quality(docs):


            print(
                "Low quality retrieval detected..."
            )


            query = self.reformulate_query(
                query
            )


            docs = self.retriever.search(
                query,
                top_k=top_k
            )



        # Reranking

        print(
            "\nReranking documents..."
        )


        ranked_docs = self.reranker.rerank(
            query,
            docs
        )



        # =================================================
        # Adaptive Score Filtering
        # =================================================


        scores = []


        for item in ranked_docs:


            if isinstance(item, dict):

                scores.append(
                    item.get(
                        "reranker_score",
                        0
                    )
                )



        selected_docs = []



        if scores:


            max_score = max(scores)


            threshold = max_score * 0.25



            print(
                "Max reranker score:",
                max_score
            )


            print(
                "Adaptive threshold:",
                threshold
            )



            for item in ranked_docs:


                if item.get(
                    "reranker_score",
                    0
                ) >= threshold:


                    selected_docs.append(item)



        else:


            selected_docs = ranked_docs[:]




        # =================================================
        # Final Context Selection
        # =================================================


        if query_type == "Simple":


            final_docs = selected_docs[:4]


        else:


            final_docs = selected_docs[:8]



        # Safety fallback

        if len(final_docs) < 2:


            final_docs = ranked_docs[:4]



        print(
            "Final selected documents:",
            len(final_docs)
        )



        return final_docs