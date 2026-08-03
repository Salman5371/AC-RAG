# =====================================================
# AC-RAG Answer Verification v5
# Improved Claim Verification
# No-Answer Handling + Better Threshold
# =====================================================


import numpy as np

from sentence_transformers import SentenceTransformer





class AnswerVerifier:



    def __init__(self):


        print(
            "Loading Answer Verification System..."
        )


        self.model = SentenceTransformer(

            "BAAI/bge-small-en-v1.5"

        )



        # Absolute hallucination indicators

        self.absolute_terms = [

            "completely",

            "completely removes",

            "eliminate",

            "eliminates",

            "always",

            "never",

            "guarantee",

            "perfectly",

            "100%"

        ]



        # Better balanced threshold

        self.support_threshold = 0.55






    # =====================================================
    # Extract Claims
    # =====================================================


    def extract_claims(

        self,

        answer

    ):


        sentences = answer.split(".")



        claims = []



        ignore_phrases = [

            "no relevant information was found",

            "no relevant information available",

            "insufficient evidence found",

            "i cannot answer",

            "not enough information"

        ]



        for sentence in sentences:



            sentence = sentence.strip()



            if len(sentence) < 10:

                continue



            skip = False



            for phrase in ignore_phrases:


                if phrase in sentence.lower():

                    skip = True



            if skip:

                continue



            claims.append(

                sentence

            )



        return claims






    # =====================================================
    # Semantic Similarity
    # =====================================================


    def semantic_score(

        self,

        claim,

        context

    ):



        claim_embedding = self.model.encode(

            claim,

            normalize_embeddings=True

        )



        context_embedding = self.model.encode(

            context,

            normalize_embeddings=True

        )



        score = np.dot(

            claim_embedding,

            context_embedding

        )


        return float(score)







    # =====================================================
    # Detect Overclaim
    # =====================================================


    def overclaim_penalty(

        self,

        claim

    ):


        penalty = 0



        lower = claim.lower()



        for term in self.absolute_terms:



            if term in lower:


                penalty += 0.20



        return penalty







    # =====================================================
    # Main Verification
    # =====================================================


    def verify(

        self,

        answer,

        documents

    ):



        print(

            "\nVerifying answer claims..."

        )



        # ---------------------------------
        # No information answer handling
        # ---------------------------------


        no_answer_patterns = [

            "no relevant information was found",

            "no relevant information available",

            "insufficient evidence found"

        ]



        for pattern in no_answer_patterns:



            if pattern in answer.lower():



                print(

                    "No-information response detected."

                )



                return {


                    "faithfulness_score": 1.0,


                    "supported_claims": 0,


                    "total_claims": 0,


                    "claim_results": [],


                    "supported": True,


                    "no_information": True

                }




        # ---------------------------------
        # Normal Verification
        # ---------------------------------



        context = " ".join(

            documents

        )



        claims = self.extract_claims(

            answer

        )



        supported_claims = 0



        results = []



        for claim in claims:



            semantic = self.semantic_score(

                claim,

                context

            )



            penalty = self.overclaim_penalty(

                claim

            )



            final_score = (

                semantic

                -

                penalty

            )



            supported = (

                final_score >= self.support_threshold

            )



            if supported:


                supported_claims += 1




            results.append(

                {


                    "claim":

                        claim,


                    "semantic_score":

                        semantic,


                    "penalty":

                        penalty,


                    "final_score":

                        final_score,


                    "supported":

                        supported

                }

            )



        if len(claims) > 0:


            faithfulness = (

                supported_claims /

                len(claims)

            )


        else:


            faithfulness = 1.0





        print(

            "Total Claims:",

            len(claims)

        )


        print(

            "Supported Claims:",

            supported_claims

        )


        print(

            "Faithfulness Score:",

            faithfulness

        )




        return {



            "faithfulness_score":

                faithfulness,


            "supported_claims":

                supported_claims,


            "total_claims":

                len(claims),



            "claim_results":

                results,



            "supported":

                faithfulness >= 0.70,


            "no_information":

                False

        }