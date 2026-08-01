# =====================================================
# AC-RAG Adaptive Retriever
# Adaptive Query Refinement + Quality Checking + Reranking
# =====================================================


import os
import sys


# Add retrieval folder to path
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


        print("Loading Adaptive Retrieval System...")


        self.retriever = retriever


        # If reranker passed from outside
        # use it, otherwise create new one

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


            elif isinstance(doc, tuple):

                text = str(
                    doc[0]
                )


            else:

                text = str(doc)



            total_length += len(text)



        avg_length = total_length / len(docs)



        print(
            "Average document length:",
            avg_length
        )



        # quality threshold

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



        # -------------------------
        # First Retrieval
        # -------------------------


        docs = self.retriever.search(
            query,
            top_k=20
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
                "\nLow quality retrieval detected..."
            )


            query = self.reformulate_query(
                query
            )



            docs = self.retriever.search(
                query,
                top_k=20
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



        # Return best documents

        return ranked_docs[:5]