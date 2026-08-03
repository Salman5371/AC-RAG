# =====================================================
# AC-RAG Context Quality Checker v2
# Evidence Verification using:
# 1. Semantic Similarity
# 2. Reranker Confidence
# 3. Keyword Evidence
# =====================================================


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
            query.lower().split()
        )


        important_words = [

            w for w in query_words

            if len(w) > 3

        ]



        if not important_words:

            return 0



        text = " ".join(
            documents
        ).lower()



        matched = 0



        for word in important_words:


            if word in text:

                matched += 1




        return matched / len(
            important_words
        )




    # =====================================================
    # Final Context Check
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



        print(
            "Semantic Score:",
            semantic
        )


        print(
            "Keyword Evidence:",
            keyword
        )



        # ---------------------------------
        # Combined Evidence Score
        # ---------------------------------


        final_score = (

            0.7 * semantic

            +

            0.3 * keyword

        )



        print(
            "Final Evidence Score:",
            final_score
        )



        # Higher threshold

        if final_score >= 0.65:

            sufficient = True


        else:

            sufficient = False



        return {


            "semantic_score":
                semantic,


            "keyword_score":
                keyword,


            "evidence_score":
                final_score,


            "sufficient":
                sufficient

        }