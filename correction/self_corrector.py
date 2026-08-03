# =====================================================
# AC-RAG Self Corrector v3
# Claim Level Correction
# =====================================================


class SelfCorrector:


    def __init__(

        self,

        retriever,

        generator,

        verifier

    ):


        print(
            "Loading Self Correction System..."
        )


        self.retriever = retriever

        self.generator = generator

        self.verifier = verifier




    # ==========================================
    # Extract Unsupported Claims
    # ==========================================


    def get_bad_claims(

        self,

        verification

    ):


        bad_claims = []


        for item in verification.get(

            "claim_results",

            []

        ):


            if not item["supported"]:


                claim = item["claim"]


                bad_claims.append(

                    claim

                )



        return bad_claims




    # ==========================================
    # Build Correction Query
    # ==========================================


    def build_query(

        self,

        claim

    ):


        return (

            "Find supporting evidence for this statement: "

            +

            claim

        )




    # ==========================================
    # Clean Generated Output
    # ==========================================


    def clean_answer(

        self,

        text

    ):


        remove_list = [

            "Answer:",

            "provided context",

            "provided evidence",

            "Insufficient evidence found.",

        ]


        for item in remove_list:


            text = text.replace(

                item,

                ""

            )


        return text.strip()




    # ==========================================
    # Correct Single Claim
    # ==========================================


    def correct_claim(

        self,

        claim

    ):


        print(

            "\nCorrecting Claim:"

        )


        print(

            claim

        )



        query = self.build_query(

            claim

        )


        print(

            "\nCorrection Query:"

        )


        print(

            query

        )



        docs = self.retriever.retrieve(

            query

        )



        context = "\n".join(

            [

                d["text"]

                for d in docs

            ]

        )



        prompt_question = (

            "Rewrite this claim using only supported evidence:\n"

            +

            claim

        )



        corrected = self.generator.generate(

            prompt_question,

            context

        )



        corrected = self.clean_answer(

            corrected

        )


        return corrected





    # ==========================================
    # Main Correction
    # ==========================================


    def correct(

        self,

        question,

        answer,

        documents

    ):



        print(

            "\nStarting Self Correction..."

        )



        verification = self.verifier.verify(

            answer,

            documents

        )



        bad_claims = self.get_bad_claims(

            verification

        )



        if len(bad_claims)==0:


            print(

                "No unsupported claims found."

            )


            return answer




        print(

            "\nUnsupported Claims:"

        )


        for claim in bad_claims:


            print(

                "-",

                claim

            )



        corrected_answer = answer



        for claim in bad_claims[:2]:


            new_claim = self.correct_claim(

                claim

            )


            corrected_answer = corrected_answer.replace(

                claim,

                new_claim

            )



        corrected_answer = self.clean_answer(

            corrected_answer

        )



        print(

            "\nClaim correction completed."

        )


        return corrected_answer