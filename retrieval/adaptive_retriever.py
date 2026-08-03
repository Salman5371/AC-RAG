# =====================================================
# AC-RAG Adaptive Retriever v6.1
# Adaptive Evidence-Aware Retrieval
#
# Features:
# - Hybrid Retrieval (FAISS + BM25)
# - Neural Reranking
# - Evidence Validation
# - Adaptive Re-retrieval
# - Debug Evidence Inspection
# =====================================================


from verification.context_checker import ContextQualityChecker

from retrieval.reranker import Reranker




class AdaptiveRetriever:



    def __init__(self, base_retriever):


        print(
            "Loading Adaptive Retrieval System..."
        )


        self.base_retriever = base_retriever


        self.reranker = Reranker()


        self.context_checker = ContextQualityChecker()



        self.last_context_status = False

        self.last_evidence_score = 0





    # =====================================================
    # Retrieve
    # =====================================================


    def retrieve(self, query):


        print(
            "\n[Adaptive Retrieval]"
        )


        print(
            "Query:",
            query
        )



        # =================================================
        # Stage 1: Hybrid Retrieval
        # =================================================


        documents = self.base_retriever.search(

            query,

            top_k=50

        )



        print(

            "Retrieved documents:",

            len(documents)

        )



        if not documents:


            self.last_context_status = False

            self.last_evidence_score = 0


            return []





        # =================================================
        # Stage 2: Neural Reranking
        # =================================================


        print(

            "\nReranking documents..."

        )


        ranked_docs = self.reranker.rerank(

            query,

            documents

        )



        selected_docs = ranked_docs[:5]



        print(

            "Initial selected documents:",

            len(selected_docs)

        )





        # Debug evidence inspection

        print(

            "\nTop Evidence Preview:"

        )


        for i, doc in enumerate(selected_docs):


            print(

                "\nDOC",

                i+1

            )


            print(

                doc["text"][:200]

            )





        # =================================================
        # Stage 3: Context Quality Checking
        # =================================================


        texts = [

            doc["text"]

            for doc in selected_docs

        ]



        quality = self.context_checker.check_context(

            query,

            texts

        )



        self.last_evidence_score = quality[

            "evidence_score"

        ]


        self.last_context_status = quality[

            "sufficient"

        ]



        print(

            "Evidence Score:",

            self.last_evidence_score

        )


        print(

            "Context Sufficient:",

            self.last_context_status

        )





        # =================================================
        # Stage 4: Adaptive Retrieval
        # =================================================


        if not self.last_context_status:


            print(

                "\nWeak evidence detected..."

            )


            print(

                "Activating adaptive retrieval..."

            )



            refined_query = (

                "Provide detailed academic information about: "

                +

                query

            )



            print(

                "Reformulated Query:",

                refined_query

            )




            more_documents = self.base_retriever.search(

                refined_query,

                top_k=80

            )



            print(

                "Retrieved documents:",

                len(more_documents)

            )



            if more_documents:


                print(

                    "\nReranking adaptive results..."

                )



                reranked = self.reranker.rerank(

                    refined_query,

                    more_documents

                )



                adaptive_docs = reranked[:8]



                print(

                    "Adaptive selected documents:",

                    len(adaptive_docs)

                )




                new_texts = [

                    doc["text"]

                    for doc in adaptive_docs

                ]




                print(

                    "\nRe-checking improved context..."

                )



                improved_quality = self.context_checker.check_context(

                    query,

                    new_texts

                )



                self.last_evidence_score = improved_quality[

                    "evidence_score"

                ]


                self.last_context_status = improved_quality[

                    "sufficient"

                ]




                if self.last_context_status:


                    print(

                        "Improved evidence accepted."

                    )


                    selected_docs = adaptive_docs



                else:


                    print(

                        "Evidence still insufficient."

                    )


                    selected_docs = adaptive_docs





        else:


            print(

                "Evidence quality acceptable."

            )





        return selected_docs