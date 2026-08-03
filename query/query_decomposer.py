# =====================================================
# AC-RAG Query Decomposer v3
# Multi Intent Query Understanding Module
# =====================================================


import re




class QueryDecomposer:



    def __init__(self):

        print(
            "Loading Query Decomposer..."
        )



    # =====================================================
    # Query Cleaning
    # =====================================================


    def clean_query(

        self,

        text

    ):


        # Remove quotes

        text = text.replace(
            '"',
            ""
        )

        text = text.replace(
            "'",
            ""
        )


        # Remove extra spaces

        text = re.sub(

            r"\s+",

            " ",

            text

        )


        text = text.strip()



        # Remove wrong punctuation

        text = re.sub(

            r"\.+",

            ".",

            text

        )


        text = re.sub(

            r"\?+",

            "?",

            text

        )


        # Fix .?

        text = text.replace(

            ".?",

            "?"

        )


        return text.strip()




    # =====================================================
    # Main Decomposition
    # =====================================================


    def decompose(

        self,

        query

    ):



        print(

            "\nAnalyzing Query..."

        )



        query = self.clean_query(

            query

        )



        queries = []



        # -------------------------------------
        # Split by question marks
        # -------------------------------------


        parts = re.split(

            r"\?+",

            query

        )



        for part in parts:



            part = part.strip()



            if len(part) < 10:

                continue



            part = self.clean_query(

                part

            )



            # Remove ending punctuation

            part = part.rstrip(

                ".?"

            )



            # Add final question mark

            part += "?"



            queries.append(

                part

            )





        # -------------------------------------
        # Handle newline separated questions
        # -------------------------------------


        if len(queries) == 1:


            lines = query.split(

                "\n"

            )


            temp = []



            for line in lines:



                line = self.clean_query(

                    line

                )


                line = line.rstrip(

                    ".?"

                )



                if len(line) > 10:


                    temp.append(

                        line + "?"

                    )



            if len(temp) > 1:


                queries = temp





        # -------------------------------------
        # Remove duplicates
        # -------------------------------------


        unique_queries = []



        for q in queries:



            if q not in unique_queries:


                unique_queries.append(

                    q

                )



        queries = unique_queries





        # -------------------------------------
        # Fallback
        # -------------------------------------


        if len(queries) == 0:


            queries = [

                query

            ]





        print(

            "Sub Queries:",

            len(queries)

        )



        for i,q in enumerate(

            queries

        ):


            print(

                f"{i+1}. {q}"

            )



        return queries