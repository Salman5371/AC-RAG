# =====================================================
# AC-RAG Quantitative Evaluation
#
# Calculates:
# - Answer Success Rate
# - Faithfulness
# - Verification Rate
# - Unknown Query Rejection
# =====================================================


import json
import os
import csv



class QuantitativeEvaluator:


    def __init__(self, result_file):


        self.result_file = result_file


        with open(
            result_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.results = json.load(f)




    # =====================================================
    # Calculate Metrics
    # =====================================================


    def calculate(self):


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


        faithfulness_scores = [

            r.get(
                "faithfulness_score",
                0
            )

            for r in self.results

        ]


        avg_faithfulness = (

            sum(
                faithfulness_scores
            )
            /
            len(
                faithfulness_scores
            )

            if faithfulness_scores

            else 0

        )



        answered = total - no_information



        answer_success = (

            answered / total * 100

            if total

            else 0

        )



        verification_rate = (

            supported / total * 100

            if total

            else 0

        )



        return {


            "Total Questions":

                total,


            "Answered Questions":

                answered,


            "No Information Responses":

                no_information,


            "Answer Success Rate (%)":

                round(
                    answer_success,
                    2
                ),


            "Verification Success (%)":

                round(
                    verification_rate,
                    2
                ),


            "Average Faithfulness":

                round(
                    avg_faithfulness,
                    4
                )


        }




    # =====================================================
    # Save Report
    # =====================================================


    def save_report(self):


        metrics = self.calculate()



        folder = os.path.dirname(

            self.result_file

        )


        csv_path = os.path.join(

            folder,

            "quantitative_report.csv"

        )


        txt_path = os.path.join(

            folder,

            "quantitative_summary.txt"

        )



        # CSV


        with open(

            csv_path,

            "w",

            newline="",

            encoding="utf-8"

        ) as f:


            writer = csv.writer(f)


            writer.writerow(

                [
                    "Metric",
                    "Value"
                ]

            )


            for k,v in metrics.items():


                writer.writerow(

                    [
                        k,
                        v
                    ]

                )



        # TXT


        with open(

            txt_path,

            "w",

            encoding="utf-8"

        ) as f:


            f.write(

                "AC-RAG Quantitative Evaluation Report\n"

            )


            f.write(

                "====================================\n\n"

            )


            for k,v in metrics.items():


                f.write(

                    f"{k}: {v}\n"

                )



        print(

            "\nEvaluation Completed"

        )


        print(

            "Saved:"

        )


        print(

            csv_path

        )


        print(

            txt_path

        )




# =====================================================
# Run
# =====================================================


if __name__ == "__main__":


    evaluator = QuantitativeEvaluator(

        "evaluation/results/run_001/benchmark_results.json"

    )


    evaluator.save_report()