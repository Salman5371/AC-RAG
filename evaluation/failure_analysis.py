# =====================================================
# AC-RAG Failure Case Analysis
#
# Analyze:
# - No Information cases
# - Verification failures
# - Category performance
# =====================================================


import json
import csv
import os



class FailureAnalyzer:


    def __init__(self, result_file):


        self.result_file = result_file


        with open(
            result_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.results = json.load(f)



    # =====================================================
    # Extract Failure Cases
    # =====================================================


    def analyze_failures(self):


        failures = []


        for item in self.results:


            question = item.get(
                "question",
                ""
            )


            answer = item.get(
                "answer",
                ""
            )


            supported = item.get(
                "supported",
                False
            )


            no_information = item.get(
                "no_information",
                False
            )


            issue = None



            if no_information:


                issue = "No Information Response"



            elif not supported:


                issue = "Verification Failure"



            if issue:


                failures.append(

                    {

                        "question":
                            question,

                        "answer":
                            answer,

                        "issue":
                            issue

                    }

                )



        return failures




    # =====================================================
    # Category Analysis
    # =====================================================


    def category_analysis(self):


        categories = {}



        for item in self.results:


            category = item.get(

                "category",

                "unknown"

            )


            if category not in categories:


                categories[category] = {


                    "total":0,

                    "answered":0,

                    "rejected":0

                }



            categories[category]["total"] += 1



            if item.get(

                "no_information",

                False

            ):


                categories[category]["rejected"] += 1


            else:


                categories[category]["answered"] += 1



        return categories




    # =====================================================
    # Save Reports
    # =====================================================


    def save_reports(self):


        folder = os.path.dirname(

            self.result_file

        )



        failures = self.analyze_failures()



        failure_file = os.path.join(

            folder,

            "failure_cases.csv"

        )



        with open(

            failure_file,

            "w",

            newline="",

            encoding="utf-8"

        ) as f:


            writer = csv.DictWriter(

                f,

                fieldnames=[

                    "question",

                    "answer",

                    "issue"

                ]

            )


            writer.writeheader()


            writer.writerows(

                failures

            )




        category_file = os.path.join(

            folder,

            "category_analysis.csv"

        )


        categories = self.category_analysis()



        with open(

            category_file,

            "w",

            newline="",

            encoding="utf-8"

        ) as f:


            writer = csv.writer(f)


            writer.writerow(

                [

                    "Category",

                    "Total",

                    "Answered",

                    "Rejected"

                ]

            )


            for k,v in categories.items():


                writer.writerow(

                    [

                        k,

                        v["total"],

                        v["answered"],

                        v["rejected"]

                    ]

                )



        print(

            "\nFailure Analysis Completed"

        )


        print(

            failure_file

        )


        print(

            category_file

        )





# =====================================================
# Run
# =====================================================


if __name__ == "__main__":



    analyzer = FailureAnalyzer(

        "evaluation/results/run_001/benchmark_results.json"

    )


    analyzer.save_reports()