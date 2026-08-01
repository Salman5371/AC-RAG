import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)



class QwenGenerator:


    def __init__(self):

        print("Loading Qwen model...")


        self.model_name = (
            "Qwen/Qwen2.5-3B-Instruct"
        )


        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )


        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
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
You are an expert AI assistant.

Answer the question using only the provided context.

Context:
{context}


Question:
{question}


Answer:
"""


        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(
            self.model.device
        )


        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True
            )


        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )


        answer = response.split(
            "Answer:"
        )[-1]


        return answer.strip()