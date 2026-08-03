from self_corrector import SelfCorrector



# ==========================
# Dummy Retriever
# ==========================

class DummyRetriever:


    def retrieve(
        self,
        query
    ):


        print(
            "\nRetrieving new evidence..."
        )


        return [

            {
                "text":
                "RAG reduces hallucination by retrieving external knowledge and providing relevant context to LLMs."
            }

        ]




# ==========================
# Dummy Generator
# ==========================

class DummyGenerator:


    def generate(
        self,
        question,
        context
    ):


        print(
            "\nGenerating corrected answer..."
        )


        return (

            "RAG reduces hallucination by "
            "using external knowledge retrieval. "
            "However, it does not completely "
            "eliminate hallucination."

        )




# ==========================
# Dummy Verifier
# ==========================


class DummyVerifier:


    def verify(
        self,
        answer,
        docs
    ):


        return {


            "supported":False,


            "claim_results":[


                {

                "claim":
                "RAG completely removes hallucination",


                "supported":
                False

                }


            ]

        }





# ==========================
# Initialize
# ==========================


corrector = SelfCorrector(

    retriever=DummyRetriever(),

    generator=DummyGenerator(),

    verifier=DummyVerifier()

)




answer = """

RAG completely removes hallucination.

"""



documents = [

"RAG reduces hallucination using external knowledge."

]



result = corrector.correct(

    "How does RAG reduce hallucination?",

    answer,

    documents

)



print(
    "\n========== FINAL ANSWER =========="
)


print(result)