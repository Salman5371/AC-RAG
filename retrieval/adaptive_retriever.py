# =====================================================
# AC-RAG Adaptive Self-Correcting Retriever
# Root Execution Compatible Version
# =====================================================


import os
import sys


ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


if ROOT_DIR not in sys.path:

    sys.path.append(ROOT_DIR)



from retrieval.reranker import Reranker

from verification.context_checker import (
    ContextQualityChecker
)




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



        self.context_checker = (
            ContextQualityChecker()
        )




    def analyze_query(
        self,
        query
    ):


        terms = [

            "methods",
            "techniques",
            "compare",
            "limitations",
            "advantages",
            "improve",
            "framework",
            "architecture",
            "process",
            "explain",
            "detailed"

        ]


        score = 0


        for term in terms:

            if term in query.lower():

                score += 1



        if len(query.split()) > 10:

            score += 1



        if score >= 2:

            return "Complex"


        return "Simple"




    def check_context_quality(
        self,
        query,
        docs
    ):


        texts = []


        for doc in docs:

            texts.append(
                doc["text"]
            )



        result = self.context_checker.check_context(

            query,

            texts

        )



        print(
            "Evidence Score:",
            result["evidence_score"]
        )


        print(
            "Context Sufficient:",
            result["sufficient"]
        )



        return result["sufficient"]




    def retrieve_documents(
        self,
        query,
        top_k
    ):


        docs = self.retriever.search(

            query,

            top_k=top_k

        )


        print(
            "Retrieved documents:",
            len(docs)
        )



        print(
            "\nReranking documents..."
        )



        ranked = self.reranker.rerank(

            query,

            docs

        )


        return ranked




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



        query_type = self.analyze_query(
            query
        )


        print(
            "Query Type:",
            query_type
        )



        if query_type == "Complex":

            top_k = 50

        else:

            top_k = 25




        ranked_docs = self.retrieve_documents(

            query,

            top_k

        )



        selected_docs = ranked_docs[:5]



        print(
            "Initial selected documents:",
            len(selected_docs)
        )



        context_ok = self.check_context_quality(

            query,

            selected_docs

        )



        if not context_ok:



            print(
                "\nWeak evidence detected..."
            )


            print(
                "Activating adaptive retrieval..."
            )



            new_query = self.reformulate_query(

                query

            )



            ranked_docs = self.retrieve_documents(

                new_query,

                top_k + 25

            )



            selected_docs = ranked_docs[:8]



            print(
                "After correction selected documents:",
                len(selected_docs)
            )



            print(
                "\nRe-checking improved context..."
            )



            retry = self.check_context_quality(

                new_query,

                selected_docs

            )



            if retry:

                print(
                    "Improved evidence accepted."
                )

            else:

                print(
                    "Evidence still insufficient."
                )



        else:


            print(
                "Evidence quality acceptable."
            )



        return selected_docs