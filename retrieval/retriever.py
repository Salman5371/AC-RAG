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


        tokenized = [
            c.lower().split()
            for c in self.chunks
        ]


        self.bm25 = BM25Okapi(
            tokenized
        )


    def search(
        self,
        query,
        top_k=10
    ):


        results={}


        # BM25

        bm25_scores = self.bm25.get_scores(
            query.lower().split()
        )


        bm25_ids = np.argsort(
            bm25_scores
        )[::-1][:top_k]


        for idx in bm25_ids:

            results[idx]=0.5



        # FAISS


        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True
        )


        scores, ids = self.index.search(
            q_emb,
            top_k
        )


        for rank,idx in enumerate(ids[0]):

            results[idx]=(
                results.get(idx,0)
                +
                1/(rank+1)
            )



        ranked = sorted(
            results.items(),
            key=lambda x:x[1],
            reverse=True
        )


        return [
            self.chunks[idx]
            for idx,score in ranked
        ]