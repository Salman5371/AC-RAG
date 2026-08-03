# =====================================================
# AC-RAG Result Logger
# Save Experiment Results
# =====================================================


import json
import csv
import os




class ResultLogger:



    def __init__(self):


        self.result_dir = "results"


        os.makedirs(

            self.result_dir,

            exist_ok=True

        )


        self.json_file = os.path.join(

            self.result_dir,

            "benchmark_results.json"

        )


        self.csv_file = os.path.join(

            self.result_dir,

            "benchmark_results.csv"

        )





    def save(

        self,

        result

    ):



        # ==========================
        # JSON Save
        # ==========================


        data = []



        if os.path.exists(

            self.json_file

        ):


            with open(

                self.json_file,

                "r",

                encoding="utf-8"

            ) as f:


                data = json.load(f)




        data.append(

            result

        )



        with open(

            self.json_file,

            "w",

            encoding="utf-8"

        ) as f:


            json.dump(

                data,

                f,

                indent=4,

                ensure_ascii=False

            )




        # ==========================
        # CSV Save
        # ==========================


        file_exists = os.path.exists(

            self.csv_file

        )



        with open(

            self.csv_file,

            "a",

            newline="",

            encoding="utf-8"

        ) as f:



            writer = csv.DictWriter(

                f,

                fieldnames=[

                    "question",

                    "answer",

                    "faithfulness_score",

                    "supported",

                    "no_information"

                ]

            )



            if not file_exists:


                writer.writeheader()



            writer.writerow(

                {


                    "question":

                        result.get(

                            "question",

                            ""

                        ),



                    "answer":

                        result.get(

                            "answer",

                            ""

                        ),



                    "faithfulness_score":

                        result.get(

                            "faithfulness_score",

                            0

                        ),



                    "supported":

                        result.get(

                            "supported",

                            False

                        ),



                    "no_information":

                        result.get(

                            "no_information",

                            False

                        )


                }

            )



        print(

            "\nResult saved successfully."

        )