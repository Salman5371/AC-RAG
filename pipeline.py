# =====================================================
# AC-RAG Main Pipeline v5
# Adaptive Self-Correcting Retrieval-Augmented Generation
#
# Features:
# - Query Decomposition
# - Adaptive Hybrid Retrieval
# - Evidence Gate
# - Qwen Generation
# - Answer Verification
# - Self Correction
# - Experiment Result Logging
# =====================================================


from retrieval.retriever import HybridRetriever
from retrieval.adaptive_retriever import AdaptiveRetriever

from generation.qwen_generator import QwenGenerator

from verification.answer_verifier import AnswerVerifier

from correction.self_corrector import SelfCorrector

from query.query_decomposer import QueryDecomposer

from evaluation.result_logger import ResultLogger




class ACRAGPipeline:



    def __init__(self):


        print(
            "\nInitializing AC-RAG System..."
        )



        # ==============================
        # Query Module
        # ==============================


        self.decomposer = QueryDecomposer()



        # ==============================
        # Retrieval Module
        # ==============================


        self.base_retriever = HybridRetriever()


        self.retriever = AdaptiveRetriever(

            self.base_retriever

        )



        # ==============================
        # Generation Module
        # ==============================


        self.generator = QwenGenerator()



        # ==============================
        # Verification Module
        # ==============================


        self.verifier = AnswerVerifier()



        # ==============================
        # Correction Module
        # ==============================


        self.corrector = SelfCorrector(

            self.retriever,

            self.generator,

            self.verifier

        )



        # ==============================
        # Logger
        # ==============================


        self.logger = ResultLogger()



        print(

            "\nAC-RAG Ready!"

        )






    # =====================================================
    # Multi Query Retrieval
    # =====================================================


    def retrieve_multi_query(

        self,

        question

    ):



        queries = self.decomposer.decompose(

            question

        )



        all_documents = []

        context_status = []



        print(

            "\n========== QUERY RETRIEVAL =========="

        )




        for query in queries:



            print(

                "\nRetrieving for:",

                query

            )



            docs = self.retriever.retrieve(

                query

            )



            all_documents.extend(

                docs

            )



            # collect evidence status

            if hasattr(

                self.retriever,

                "last_context_status"

            ):


                context_status.append(

                    self.retriever.last_context_status

                )





        # Remove duplicate documents


        unique_docs = {}



        for doc in all_documents:



            key = doc.get(

                "chunk_id",

                doc["text"][:100]

            )



            if key not in unique_docs:


                unique_docs[key] = doc





        documents = list(

            unique_docs.values()

        )



        print(

            "\nTotal Unique Documents:",

            len(documents)

        )




        # Final evidence decision


        context_valid = True



        if context_status:


            context_valid = all(

                context_status

            )




        return {


            "documents":

                documents,


            "context_valid":

                context_valid

        }







    # =====================================================
    # Context Builder
    # =====================================================


    def build_context(

        self,

        documents

    ):



        if not documents:

            return ""



        return "\n\n".join(

            [

                doc["text"]

                for doc in documents

            ]

        )







    # =====================================================
    # Main Pipeline
    # =====================================================


    def run(

        self,

        question

    ):



        print(

            "\n=============================="

        )


        print(

            "QUESTION:"

        )


        print(

            question

        )


        print(

            "=============================="

        )




        # ==============================
        # Retrieval
        # ==============================



        retrieval_result = self.retrieve_multi_query(

            question

        )



        documents = retrieval_result["documents"]



        context_valid = retrieval_result["context_valid"]



        context = self.build_context(

            documents

        )





        # ==============================
        # Evidence Gate
        # ==============================



        if not context_valid:



            answer = (

                "No relevant information was found."

            )



            print(

                "\nEvidence insufficient."

            )


            print(

                "\nGenerated Answer:"

            )


            print(

                answer

            )



            verification = {


                "faithfulness_score":

                    1.0,


                "supported":

                    True,


                "no_information":

                    True

            }



            self.logger.save(

                {


                    "question":

                        question,


                    "answer":

                        answer,


                    "faithfulness_score":

                        1.0,


                    "supported":

                        True,


                    "no_information":

                        True

                }

            )



            return answer







        # ==============================
        # Generation
        # ==============================



        answer = self.generator.generate(

            question,

            context

        )



        print(

            "\nGenerated Answer:"

        )


        print(

            answer

        )






        # ==============================
        # Verification
        # ==============================



        verification = self.verifier.verify(

            answer,

            [

                context

            ]

        )



        print(

            "\nVerification Result:"

        )


        print(

            verification

        )



        final_verification = verification






        # ==============================
        # Self Correction
        # ==============================



        if not verification.get(

            "supported",

            False

        ):



            print(

                "\n========== SELF CORRECTION =========="

            )



            answer = self.corrector.correct(

                question,

                answer,

                [

                    context

                ]

            )



            print(

                "\nCorrected Answer:"

            )


            print(

                answer

            )



            final_verification = self.verifier.verify(

                answer,

                [

                    context

                ]

            )



            print(

                "\nFinal Verification:"

            )


            print(

                final_verification

            )



        else:



            print(

                "\nAnswer verified successfully."

            )







        # ==============================
        # Save Result
        # ==============================



        self.logger.save(

            {


                "question":

                    question,


                "answer":

                    answer,


                "faithfulness_score":

                    final_verification.get(

                        "faithfulness_score",

                        0

                    ),


                "supported":

                    final_verification.get(

                        "supported",

                        False

                    ),


                "no_information":

                    final_verification.get(

                        "no_information",

                        False

                    )

            }

        )



        return answer






# =====================================================
# Single Query Mode
# =====================================================


if __name__ == "__main__":



    system = ACRAGPipeline()



    question = input(

        "\nEnter Question: "

    )



    final_answer = system.run(

        question

    )



    print(

        "\n========== FINAL OUTPUT =========="

    )


    print(

        final_answer

    )