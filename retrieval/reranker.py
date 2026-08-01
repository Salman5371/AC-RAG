from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        print("Loading reranker model...")

        self.model = CrossEncoder(
            "BAAI/bge-reranker-base"
        )


    def rerank(
        self,
        query,
        documents
    ):

        pairs = []


        for doc in documents:

            pairs.append(
                [
                    query,
                    doc
                ]
            )


        scores = self.model.predict(
            pairs
        )


        results = sorted(
            zip(
                documents,
                scores
            ),
            key=lambda x:x[1],
            reverse=True
        )


        return results