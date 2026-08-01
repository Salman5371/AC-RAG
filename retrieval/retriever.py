import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi



class HybridRetriever:


    def __init__(self):

        print("Loading FAISS...")


        self.index = faiss.read_index(
            "embeddings_v2/faiss.index"
        )


        with open(
            "embeddings_v2/chunks.pkl",
            "rb"
        ) as f:

            self.chunks = pickle.load(f)


        print(
            "Chunks:",
            len(self.chunks)
        )


        print(
            "Loading embedding model..."
        )


        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )



        tokenized_chunks = [
            c.lower().split()
            for c in self.chunks
        ]


        self.bm25 = BM25Okapi(
            tokenized_chunks
        )



    # =====================================
    # Hybrid Search
    # =====================================


    def search(
        self,
        query,
        top_k=10
    ):


        results = {}



        # -----------------------------
        # BM25 Retrieval
        # -----------------------------


        bm25_scores = self.bm25.get_scores(
            query.lower().split()
        )


        bm25_ids = np.argsort(
            bm25_scores
        )[::-1][:top_k]



        for rank, idx in enumerate(bm25_ids):

            results[idx] = {

                "bm25_score":
                    float(bm25_scores[idx]),

                "hybrid_score":
                    0.5

            }



        # -----------------------------
        # FAISS Retrieval
        # -----------------------------


        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )


        faiss_scores, faiss_ids = self.index.search(
            query_embedding,
            top_k
        )



        for rank, idx in enumerate(faiss_ids[0]):


            if idx not in results:

                results[idx] = {

                    "bm25_score":0.0,

                    "hybrid_score":0.0

                }



            results[idx]["faiss_score"] = float(
                faiss_scores[0][rank]
            )


            results[idx]["hybrid_score"] += (
                1/(rank+1)
            )



        # -----------------------------
        # Sort Results
        # -----------------------------


        ranked = sorted(
            results.items(),
            key=lambda x:x[1]["hybrid_score"],
            reverse=True
        )



        # -----------------------------
        # Return Metadata
        # -----------------------------


        final_results = []


        for idx, metadata in ranked:


            final_results.append(

                {

                    "chunk_id": int(idx),

                    "text": self.chunks[idx],

                    "bm25_score":
                        metadata.get(
                            "bm25_score",
                            0
                        ),

                    "faiss_score":
                        metadata.get(
                            "faiss_score",
                            0
                        ),

                    "hybrid_score":
                        metadata.get(
                            "hybrid_score",
                            0
                        )

                }

            )



        return final_results