# =====================================================
# AC-RAG Query Decomposer v4.1
# Multi Intent + Query Expansion Module
#
# Improvements:
# - Better Question Separation
# - Sentence Boundary Detection
# - Controlled Query Expansion
# - Duplicate Removal
# =====================================================


import re




class QueryDecomposer:



    def __init__(self):

        print(
            "Loading Query Decomposer..."
        )


        self.expansion_terms = {


            "components": [

                "architecture",
                "modules",
                "pipeline",
                "framework",
                "workflow"

            ],


            "retrieval augmented generation": [

                "retrieval",
                "generation",
                "augmentation"

            ],


            "hallucination": [

                "factual errors",
                "incorrect answers",
                "faithfulness"

            ],


            "verification": [

                "validation",
                "fact checking",
                "answer checking"

            ],


            "self-correction": [

                "feedback",
                "error correction",
                "refinement"

            ]

        }






    # =====================================================
    # Query Cleaning
    # =====================================================


    def clean_query(self, text):


        text = text.replace(
            '"',
            ""
        )


        text = text.replace(
            "'",
            ""
        )


        # Fix missing spaces after punctuation

        text = re.sub(

            r"\?(?=[A-Za-z])",

            "? ",

            text

        )


        text = re.sub(

            r"\.(?=[A-Z])",

            ". ",

            text

        )



        text = re.sub(

            r"\s+",

            " ",

            text

        )


        text = text.strip()



        text = re.sub(

            r"\?+",

            "?",

            text

        )


        text = re.sub(

            r"\.+",

            ".",

            text

        )



        text = text.replace(

            ".?",

            "?"

        )


        return text.strip()







    # =====================================================
    # Query Expansion
    # =====================================================


    def expand_query(self, query):


        lower_query = query.lower()


        added_terms = []



        for key, values in self.expansion_terms.items():


            if key in lower_query:


                added_terms.extend(

                    values

                )



        # Special handling for RAG abbreviation


        if re.search(

            r"\brag\b",

            lower_query

        ):


            added_terms.extend(

                [

                    "retrieval",

                    "generation",

                    "augmentation"

                ]

            )



        if added_terms:


            # remove duplicates

            added_terms = list(

                dict.fromkeys(

                    added_terms

                )

            )


            expansion = ", ".join(

                added_terms

            )


            query = (

                query.rstrip("?")

                +

                " including "

                +

                expansion

                +

                "?"

            )



        return query







    # =====================================================
    # Main Decomposition
    # =====================================================


    def decompose(self, query):


        print(

            "\nAnalyzing Query..."

        )



        query = self.clean_query(

            query

        )



        queries = []



        # Split by question marks


        parts = re.split(

            r"\?",

            query

        )



        for part in parts:


            part = part.strip()



            if len(part) < 10:

                continue



            part = part.rstrip(

                ". "

            )



            part += "?"



            queries.append(

                part

            )







        # =================================================
        # Sentence based split
        # =================================================


        if len(queries) == 1:


            sentences = re.split(

                r"(?<=[.!?])\s+(?=[A-Z])",

                query

            )


            temp = []



            for sentence in sentences:


                sentence = sentence.strip()



                if len(sentence) > 10:


                    sentence = sentence.rstrip(

                        ".?"

                    )


                    temp.append(

                        sentence + "?"

                    )



            if len(temp) > 1:


                queries = temp







        # =================================================
        # Newline split
        # =================================================


        if len(queries) == 1:


            lines = query.split(

                "\n"

            )


            temp = []



            for line in lines:


                line = line.strip()



                if len(line) > 10:


                    temp.append(

                        line.rstrip(".?") + "?"

                    )



            if len(temp) > 1:


                queries = temp







        # =================================================
        # Remove duplicates
        # =================================================


        unique_queries = []



        for q in queries:


            if q not in unique_queries:


                unique_queries.append(q)



        queries = unique_queries






        if not queries:


            queries = [

                query

            ]







        # =================================================
        # Expansion Layer
        # =================================================


        expanded_queries = []



        for q in queries:


            expanded_queries.append(

                self.expand_query(q)

            )







        print(

            "Sub Queries:",

            len(expanded_queries)

        )



        for i,q in enumerate(expanded_queries):


            print(

                f"{i+1}. {q}"

            )



        return expanded_queries