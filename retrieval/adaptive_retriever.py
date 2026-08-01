from reranker import Reranker


class AdaptiveRetriever:


    def __init__(
        self,
        retriever,
        reranker
    ):

        print("Loading system...")

        self.retriever = retriever

        self.reranker = reranker



    # ==================================
    # Retrieval Quality Checking
    # ==================================

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

                text = doc


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



    # ==================================
    # Query Reformulation
    # ==================================

    def reformulate_query(
        self,
        query
    ):


        new_query = (
            "Provide detailed information about: "
            + query
        )


        print(
            "\nReformulated Query:"
        )

        print(
            new_query
        )


        return new_query



    # ==================================
    # Adaptive Retrieval Pipeline
    # ==================================

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



        # -------------------------------
        # First Retrieval
        # -------------------------------

        docs = self.retriever.search(
            query,
            top_k=20
        )



        print(
            "\nRetrieved documents:",
            len(docs)
        )



        # -------------------------------
        # Quality Check
        # -------------------------------

        quality = self.check_quality(
            docs
        )



        if not quality:


            print(
                "\nLow quality retrieval detected"
            )


            query = self.reformulate_query(
                query
            )



            docs = self.retriever.search(
                query,
                top_k=20
            )



        # -------------------------------
        # Cross Encoder Reranking
        # -------------------------------


        print(
            "\nReranking documents..."
        )


        ranked_docs = self.reranker.rerank(
            query,
            docs
        )



        # return top 5

        return ranked_docs[:5]