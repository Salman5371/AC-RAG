# =====================================================
# AC-RAG Adaptive Retriever
# Adaptive Query Refinement + Quality Checking
# Adaptive Reranking Threshold
# =====================================================


import os
import sys


# Add retrieval folder path

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.append(
    CURRENT_DIR
)


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



        if reranker is not None:

            self.reranker = reranker

        else:

            self.reranker = Reranker()



    # =================================================
    # Retrieval Quality Checking
    # =================================================


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
            total_length / len(docs)
        )


        print(
            "Average document length:",
            avg_length
        )



        if avg_length > 200:

            return True


        return False




    # =================================================
    # Query Reformulation
    # =================================================


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
            "\nReformulated Query:",
            new_query
        )


        return new_query




    # =================================================
    # Adaptive Threshold
    # =================================================


    def adaptive_threshold(
        self,
        ranked_docs
    ):


        if not ranked_docs:

            return 0



        scores = []



        for doc in ranked_docs:


            scores.append(
                doc.get(
                    "reranker_score",
                    0
                )
            )



        max_score = max(
            scores
        )



        # Threshold based on strongest evidence

        threshold = (
            max_score * 0.35
        )



        # Minimum safety threshold

        if threshold < 0.10:

            threshold = 0.10



        print(
            "Maximum reranker score:",
            max_score
        )


        print(
            "Adaptive threshold:",
            threshold
        )


        return threshold




    # =================================================
    # Adaptive Retrieval Pipeline
    # =================================================


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



        # ---------------------------------
        # Initial Retrieval
        # ---------------------------------


        docs = self.retriever.search(
            query,
            top_k=20
        )



        print(
            "Retrieved documents:",
            len(docs)
        )



        # ---------------------------------
        # Quality Check
        # ---------------------------------


        quality = self.check_quality(
            docs
        )



        if not quality:


            print(
                "\nLow quality retrieval detected..."
            )


            query = self.reformulate_query(
                query
            )



            docs = self.retriever.search(
                query,
                top_k=20
            )



        # ---------------------------------
        # Reranking
        # ---------------------------------


        print(
            "\nReranking documents..."
        )


        ranked_docs = self.reranker.rerank(
            query,
            docs
        )



        # ---------------------------------
        # Adaptive Filtering
        # ---------------------------------


        threshold = self.adaptive_threshold(
            ranked_docs
        )



        filtered_docs = []



        for doc in ranked_docs:


            score = doc.get(
                "reranker_score",
                0
            )



            if score >= threshold:

                filtered_docs.append(
                    doc
                )



        print(
            "Documents after filtering:",
            len(filtered_docs)
        )



        # ---------------------------------
        # Safety fallback
        # ---------------------------------


        if len(filtered_docs) == 0:


            filtered_docs = ranked_docs[:2]



        return filtered_docs[:5]