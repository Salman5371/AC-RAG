# =====================================================
# AC-RAG Result Logger v2
#
# Features:
# - Automatic experiment run tracking
# - No overwrite
# - CSV + JSON + TXT saving
# - Research experiment friendly
# =====================================================


import os
import json
import csv
from datetime import datetime




class ResultLogger:



    def __init__(self):


        print(
            "Initializing Result Logger..."
        )


        self.base_folder = (

            "evaluation/results"

        )


        os.makedirs(

            self.base_folder,

            exist_ok=True

        )


        self.run_folder = self.create_run_folder()



        self.results = []



        print(

            "Saving results to:",

            self.run_folder

        )





    # =====================================================
    # Create New Run Folder
    # =====================================================


    def create_run_folder(self):


        existing = [

            folder

            for folder in os.listdir(

                self.base_folder

            )

            if folder.startswith("run_")

        ]


        run_numbers = []


        for folder in existing:


            try:

                number = int(

                    folder.split("_")[1]

                )

                run_numbers.append(number)


            except:

                pass



        if run_numbers:


            next_run = max(run_numbers) + 1


        else:

            next_run = 1



        folder_name = (

            f"run_{next_run:03d}"

        )


        path = os.path.join(

            self.base_folder,

            folder_name

        )


        os.makedirs(

            path,

            exist_ok=True

        )


        return path





    # =====================================================
    # Save Single Result
    # =====================================================


    def save(self, result):


        self.results.append(

            result

        )


        self.save_json()

        self.save_csv()

        self.save_summary()




    # =====================================================
    # Save JSON
    # =====================================================


    def save_json(self):


        path = os.path.join(

            self.run_folder,

            "benchmark_results.json"

        )


        with open(

            path,

            "w",

            encoding="utf-8"

        ) as f:


            json.dump(

                self.results,

                f,

                indent=4,

                ensure_ascii=False

            )





    # =====================================================
    # Save CSV
    # =====================================================


    def save_csv(self):


        path = os.path.join(

            self.run_folder,

            "benchmark_results.csv"

        )


        if not self.results:

            return



        keys = self.results[0].keys()



        with open(

            path,

            "w",

            newline="",

            encoding="utf-8"

        ) as f:


            writer = csv.DictWriter(

                f,

                fieldnames=keys

            )


            writer.writeheader()


            writer.writerows(

                self.results

            )





    # =====================================================
    # Save Summary
    # =====================================================


    def save_summary(self):


        path = os.path.join(

            self.run_folder,

            "summary_report.txt"

        )


        total = len(

            self.results

        )


        supported = sum(

            1

            for r in self.results

            if r.get(

                "supported",

                False

            )

        )


        no_information = sum(

            1

            for r in self.results

            if r.get(

                "no_information",

                False

            )

        )



        faithfulness = []


        for r in self.results:


            score = r.get(

                "faithfulness_score",

                None

            )


            if score is not None:

                faithfulness.append(

                    score

                )



        avg_faithfulness = (

            sum(faithfulness)

            /

            len(faithfulness)

            if faithfulness

            else 0

        )




        with open(

            path,

            "w",

            encoding="utf-8"

        ) as f:



            f.write(

                "AC-RAG Benchmark Report\n"

            )


            f.write(

                "=======================\n\n"

            )


            f.write(

                f"Run Folder: {self.run_folder}\n\n"

            )


            f.write(

                f"Total Questions: {total}\n"

            )


            f.write(

                f"Supported Answers: {supported}\n"

            )


            f.write(

                f"No Information Responses: {no_information}\n"

            )


            f.write(

                f"Average Faithfulness: {avg_faithfulness:.4f}\n"

            )


            f.write(

                f"Completed Time: {datetime.now()}\n"

            )