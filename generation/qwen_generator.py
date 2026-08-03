# =====================================================
# AC-RAG Qwen Generator
# Grounded Academic Answer Generation
# =====================================================


import torch


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)



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



        print(
            "Qwen loaded successfully"
        )




    def generate(
        self,
        question,
        context
    ):



        prompt = f"""

You are an expert research assistant specialized in
Retrieval-Augmented Generation (RAG).

Your task is to answer the user's question using ONLY
the provided context.

Strict rules:

1. Do not use external knowledge.
2. Do not invent information.
3. If the context does not contain the answer, clearly state:
   "The provided context does not contain sufficient information."

4. Write a clear academic answer.
5. Organize the answer with:
   - Definition / Introduction
   - Key explanation
   - Important points (if applicable)

Retrieved Context:
------------------

{context}

------------------


Question:

{question}


Final Answer:

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

                max_new_tokens=512,

                temperature=0.1,

                do_sample=False,

                repetition_penalty=1.1

            )




        response = self.tokenizer.decode(

            outputs[0],

            skip_special_tokens=True

        )




        answer = response.split(

            "Final Answer:"

        )[-1]



        return answer.strip()