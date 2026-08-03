# =====================================================
# AC-RAG Context Quality Checker v6
# Evidence-Aware Retrieval Validation
#
# Features:
# 1. Semantic Similarity
# 2. Keyword Evidence
# 3. Critical Concept Validation
# 4. Query Concept Coverage ⭐ NEW
# 5. Adaptive Evidence Decision
# =====================================================


import re
import numpy as np


from sentence_transformers import SentenceTransformer





class ContextQualityChecker:



    def __init__(self):


        print(

            "Loading Context Quality Checker..."

        )


        self.model = SentenceTransformer(

            "BAAI/bge-small-en-v1.5"

        )





    # =====================================================
    # Semantic Similarity
    # =====================================================


    def semantic_score(

        self,

        query,

        documents

    ):


        if not documents:

            return 0



        query_embedding = self.model.encode(

            query,

            normalize_embeddings=True

        )



        doc_embeddings = self.model.encode(

            documents,

            normalize_embeddings=True

        )



        scores = np.dot(

            doc_embeddings,

            query_embedding

        )



        return float(

            np.mean(scores)

        )







    # =====================================================
    # Keyword Evidence
    # =====================================================


    def keyword_score(

        self,

        query,

        documents

    ):


        query_words = set(

            re.findall(

                r"\w+",

                query.lower()

            )

        )



        important_words = [

            w

            for w in query_words

            if len(w) > 3

        ]



        if not important_words:

            return 0



        context = " ".join(

            documents

        ).lower()



        matched = 0



        for word in important_words:


            if word in context:

                matched += 1




        return matched / len(

            important_words

        )







    # =====================================================
    # Critical Concept Validation
    # =====================================================


    def critical_concept_score(

        self,

        query,

        documents

    ):


        words = [

            w.lower()

            for w in re.findall(

                r"\w+",

                query

            )

            if len(w) > 5

        ]



        if not words:

            return 1.0



        context = " ".join(

            documents

        ).lower()



        matched = 0



        for word in words:


            if word in context:

                matched += 1



        return matched / len(words)







    # =====================================================
    # Query Concept Coverage ⭐ NEW
    # =====================================================


    def concept_coverage(

        self,

        query,

        documents

    ):


        """

        Checks whether important concepts
        from query actually appear in context.

        Prevents:

        RAG + human emotion
        retrieving only RAG papers

        """



        query_terms = set(

            re.findall(

                r"\w+",

                query.lower()

            )

        )



        # Remove common words


        stop_words = {


            "what",

            "which",

            "where",

            "when",

            "does",

            "do",

            "the",

            "is",

            "are",

            "a",

            "an",

            "in",

            "on",

            "of",

            "for",

            "and",

            "with",

            "to",

            "how",

            "can"

        }



        concepts = [

            w

            for w in query_terms

            if w not in stop_words

            and len(w) > 4

        ]



        if not concepts:

            return 1.0



        context = " ".join(

            documents

        ).lower()



        matched = 0



        for concept in concepts:


            if concept in context:

                matched += 1



        return matched / len(

            concepts

        )








    # =====================================================
    # Final Context Decision
    # =====================================================


    def check_context(

        self,

        query,

        documents

    ):


        print(

            "\nChecking context quality..."

        )



        semantic = self.semantic_score(

            query,

            documents

        )



        keyword = self.keyword_score(

            query,

            documents

        )



        critical = self.critical_concept_score(

            query,

            documents

        )



        coverage = self.concept_coverage(

            query,

            documents

        )





        print(

            "Semantic Score:",

            semantic

        )


        print(

            "Keyword Evidence:",

            keyword

        )


        print(

            "Critical Concept Score:",

            critical

        )


        print(

            "Concept Coverage Score:",

            coverage

        )







        # =================================================
        # Combined Evidence Score
        # =================================================


        final_score = (

            0.55 * semantic

            +

            0.20 * keyword

            +

            0.15 * critical

            +

            0.10 * coverage

        )



        print(

            "Final Evidence Score:",

            final_score

        )







        # =================================================
        # Decision Logic
        # =================================================


        sufficient = False




        # Strong evidence


        if (

            semantic >= 0.72

            and

            coverage >= 0.35

            and

            keyword >= 0.20

        ):


            sufficient = True






        # Moderate evidence


        elif (

            final_score >= 0.65

            and

            coverage >= 0.30

        ):


            sufficient = True







        # Protection against semantic-only match


        if (

            coverage < 0.20

        ):


            sufficient = False






        return {


            "semantic_score":

                semantic,


            "keyword_score":

                keyword,


            "critical_concept_score":

                critical,


            "concept_coverage_score":

                coverage,


            "evidence_score":

                final_score,


            "sufficient":

                sufficient

        }