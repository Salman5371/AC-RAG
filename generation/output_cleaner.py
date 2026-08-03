# =====================================================
# AC-RAG Output Cleaner v2
# Remove Prompt Leakage and Model Identity Leakage
# =====================================================



class OutputCleaner:



    def __init__(self):

        print(
            "Loading Output Cleaner..."
        )




    def clean(

        self,

        text

    ):


        remove_patterns = [

            # Prompt leakage

            "Answer:",

            "Response:",

            "Context:",

            "Evidence:",


            # Instruction leakage

            "provided context",

            "provided evidence",

            "using only the provided evidence",

            "Answer the question using ONLY the provided evidence.",


            # Wrong fallback phrase

            "Insufficient evidence found.",


            # Model identity leakage

            "AC-RAG",

            "AC RAG",

            "ACRAG"

        ]



        for pattern in remove_patterns:



            text = text.replace(

                pattern,

                ""

            )



        # Remove extra spaces

        text = " ".join(

            text.split()

        )



        # Remove spaces before punctuation

        text = text.replace(

            " .",

            "."

        )


        text = text.replace(

            " ,",

            ","

        )



        # Final cleanup

        text = text.strip()



        return text