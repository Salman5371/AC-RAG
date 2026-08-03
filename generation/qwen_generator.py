# =====================================================
# AC-RAG Qwen Generator v4
# Clean Academic Generation
# =====================================================


import torch


from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM

)


from generation.output_cleaner import OutputCleaner




class QwenGenerator:



    def __init__(self):


        print(

            "Loading Qwen model..."

        )



        self.model_name = (

            "Qwen/Qwen2.5-3B-Instruct"

        )



        self.tokenizer = AutoTokenizer.from_pretrained(

            self.model_name

        )



        self.model = AutoModelForCausalLM.from_pretrained(

            self.model_name,

            dtype=torch.float16,

            device_map="auto"

        )



        self.cleaner = OutputCleaner()



        print(

            "Qwen loaded successfully"

        )




    # =================================================
    # Generate Answer
    # =================================================


    def generate(

        self,

        question,

        context

    ):



        prompt = f"""

You are AC-RAG, a trustworthy academic AI assistant.

Generate a concise answer based ONLY on the evidence.

Rules:

- Answer directly.
- Do not mention documents, context, or evidence.
- Do not explain your reasoning process.
- Do not repeat the question.
- Do not add unsupported information.
- Do not use absolute claims such as completely, always, never.
- Maximum length: 2 short paragraphs.

If the information is unavailable, respond only:

No relevant information was found.


Information:

{context}


Question:

{question}


Response:

"""



        inputs = self.tokenizer(

            prompt,

            return_tensors="pt",

            truncation=True,

            max_length=4096

        ).to(

            self.model.device

        )



        with torch.no_grad():


            outputs = self.model.generate(

                **inputs,

                max_new_tokens=220,

                do_sample=False,

                repetition_penalty=1.15,

                pad_token_id=self.tokenizer.eos_token_id

            )



        response = self.tokenizer.decode(

            outputs[0],

            skip_special_tokens=True

        )



        answer = response.split(

            "Response:"

        )[-1]



        answer = self.cleaner.clean(

            answer

        )



        return answer