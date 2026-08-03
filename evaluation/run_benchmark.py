# =====================================================
# AC-RAG Benchmark Runner v2
# Automated Research Evaluation Script
# =====================================================


import os
import sys
import json



# =====================================================
# Add Project Root Path
# =====================================================


PROJECT_ROOT = os.path.dirname(

    os.path.dirname(

        os.path.abspath(__file__)

    )

)


sys.path.append(

    PROJECT_ROOT

)



from pipeline import ACRAGPipeline





class BenchmarkRunner:



    def __init__(self):


        print(

            "\nInitializing AC-RAG Benchmark..."

        )



        self.system = ACRAGPipeline()



        self.questions = []



        self.results = []



        self.total = 0


        self.completed = 0


        self.failed = 0



        self.faithfulness_scores = []



        self.supported_count = 0


        self.no_information_count = 0





    # =====================================================
    # Load Benchmark Dataset
    # =====================================================


    def load_questions(self):


        path = os.path.join(

            PROJECT_ROOT,

            "evaluation",

            "benchmark.json"

        )



        with open(

            path,

            "r",

            encoding="utf-8"

        ) as f:


            self.questions = json.load(f)



        print(

            "\nTotal Questions:",

            len(self.questions)

        )







    # =====================================================
    # Run Benchmark
    # =====================================================


    def run(self):


        self.load_questions()



        for item in self.questions:



            self.total += 1



            print(

                "\n================================"

            )


            print(

                f"Question {self.total}/{len(self.questions)}"

            )


            print(

                item["question"]

            )


            print(

                "================================"

            )



            try:



                answer = self.system.run(

                    item["question"]

                )



                result = {


                    "id":

                        item["id"],



                    "category":

                        item["category"],



                    "question":

                        item["question"],



                    "answer":

                        answer

                }



                self.results.append(

                    result

                )



                self.completed += 1



                print(

                    "\nCompleted Successfully"

                )




            except Exception as e:



                print(

                    "\nError:",

                    e

                )



                self.failed += 1





        self.save_summary()






    # =====================================================
    # Save Summary
    # =====================================================


    def save_summary(self):


        os.makedirs(

            os.path.join(

                PROJECT_ROOT,

                "results"

            ),

            exist_ok=True

        )



        result_file = os.path.join(

            PROJECT_ROOT,

            "results",

            "benchmark_execution.json"

        )



        with open(

            result_file,

            "w",

            encoding="utf-8"

        ) as f:


            json.dump(

                self.results,

                f,

                indent=4,

                ensure_ascii=False

            )





        summary_file = os.path.join(

            PROJECT_ROOT,

            "results",

            "summary_report.txt"

        )




        with open(

            summary_file,

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

                f"Total Questions: {self.total}\n"

            )


            f.write(

                f"Completed: {self.completed}\n"

            )


            f.write(

                f"Failed: {self.failed}\n"

            )



        print(

            "\n================================"

        )


        print(

            "Benchmark Completed"

        )


        print(

            f"Completed: {self.completed}/{self.total}"

        )


        print(

            "\nSaved Files:"

        )


        print(

            "results/benchmark_execution.json"

        )


        print(

            "results/summary_report.txt"

        )







# =====================================================
# Main
# =====================================================


if __name__ == "__main__":



    runner = BenchmarkRunner()



    runner.run()