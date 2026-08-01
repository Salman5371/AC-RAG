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


            # New metadata format

            if isinstance(doc, dict):

                text = doc["text"]

            else:

                text = doc



            pairs.append(
                [
                    query,
                    text
                ]
            )



        scores = self.model.predict(
            pairs
        )



        results = []



        for doc, score in zip(
            documents,
            scores
        ):


            if isinstance(doc, dict):

                doc["reranker_score"] = float(
                    score
                )

                results.append(
                    doc
                )


            else:

                results.append(
                    {
                        "text": doc,
                        "reranker_score": float(score)
                    }
                )



        results = sorted(
            results,
            key=lambda x:x["reranker_score"],
            reverse=True
        )


        return results