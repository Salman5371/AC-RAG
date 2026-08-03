# =====================================================
# AC-RAG Output Cleaner v5
# Final Generation Cleanup
# =====================================================


class OutputCleaner:


    def __init__(self):

        print("Loading Output Cleaner...")



    def clean(self, text):


        if not text:

            return ""



        text = text.strip()



        lower_text = text.lower()



        # =========================================
        # Unknown Answer Hard Stop
        # =========================================


        unknown_patterns = [

            "no relevant information was found",

            "no relevant information available",

            "insufficient evidence found"

        ]



        for pattern in unknown_patterns:


            if pattern in lower_text:


                return "No relevant information was found."





        # =========================================
        # Remove instruction leakage blocks
        # =========================================


        leakage_patterns = [

            "Directly answering without mentioning any irrelevant information.",

            "Directly addressing the question while adhering strictly to provided guidelines.",

            "Response should fit within 2 short paragraphs maximum.",

            "To generate a concise answer based solely on the provided information.",

            "Based solely on the provided information.",

            "Based solely on provided information.",

            "According to given information.",

            "Your request cannot be answered based solely on the provided information.",

            "Answer the question using ONLY the provided evidence.",

            "using only the provided evidence",


            "AC-RAG",

            "AC RAG",

            "ACRAG"

        ]



        for pattern in leakage_patterns:


            text = text.replace(

                pattern,

                ""

            )





        # =========================================
        # Remove duplicated instruction sentences
        # =========================================


        bad_start_words = [

            "Instructions:",

            "Rules:",

            "Answering guidelines:",

            "Generation rules:"

        ]



        for word in bad_start_words:


            if word in text:


                text = text.split(word)[0]





        # =========================================
        # Clean spacing
        # =========================================


        text = " ".join(

            text.split()

        )



        return text.strip()