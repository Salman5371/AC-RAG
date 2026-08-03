# =====================================================
# AC-RAG Qwen Generator v7.2
# Evidence-Grounded Academic Generation
#
# Improvements:
# - Hallucination control
# - Better yes/no reasoning
# - Component question control
# - Reduced unnecessary expansion
# - Evidence-only generation
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







    # =====================================================
    # Generate Answer
    # =====================================================


    def generate(

        self,

        question,

        context

    ):



        if not context.strip():


            return (

                "No relevant information was found."

            )






        messages = [



            {


                "role":

                "system",



                "content":

                """

You are an academic AI assistant specialized in evidence-grounded question answering.



Your task is to answer ONLY using the provided information.



========================
STRICT EVIDENCE RULES
========================


1. Answer the user question directly.

2. Use only facts explicitly supported by the provided information.

3. Do not introduce external knowledge, examples, comparisons, or research findings.

4. Do not add technical mechanisms unless they are clearly mentioned in the evidence.

5. Do not mention documents, context, retrieval process, AC-RAG, or internal system details.

6. Do not explain your reasoning process.

7. If evidence is insufficient, clearly state that the information is not available.

8. Do not convert similarity into certainty.



========================
QUESTION TYPE RULES
========================


For component/architecture questions:


- Provide only the requested components and their basic functions.

- Do not add advanced implementation details.

- Do not add speculative mechanisms.



For definition questions:


- Give a concise definition first.

- Add only directly supported explanation.



For yes/no questions:


- Start with Yes or No only if evidence supports it.

- Do not accept false assumptions in the question.

- For words such as:
  "always",
  "guarantee",
  "100%",
  "completely",

  explain limitations carefully.



========================
HALLUCINATION CONTROL
========================


- RAG may reduce hallucination but does not guarantee complete elimination.

- Do not claim certainty beyond the evidence.

- Do not add information about other technologies or applications unless explicitly provided.



========================
ANSWER STYLE
========================


- Simple questions: maximum 100 words.

- Technical questions: maximum 180 words.

- Use concise academic language.

- Focus only on answering the asked question.



"""

            },



            {


                "role":

                "user",



                "content":


                f"""

Information:


{context}



Question:


{question}



Answer:

"""

            }


        ]







        prompt = self.tokenizer.apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True

        )







        inputs = self.tokenizer(

            prompt,

            return_tensors="pt",

            truncation=True,

            max_length=4096

        ).to(

            self.model.device

        )







        input_length = inputs.input_ids.shape[1]







        with torch.no_grad():



            outputs = self.model.generate(

                **inputs,

                max_new_tokens=180,

                do_sample=False,

                repetition_penalty=1.15,

                pad_token_id=self.tokenizer.eos_token_id

            )







        generated_tokens = outputs[

            0

        ][

            input_length:

        ]







        answer = self.tokenizer.decode(

            generated_tokens,

            skip_special_tokens=True

        )







        answer = self.cleaner.clean(

            answer

        )







        if not answer:


            answer = (

                "No relevant information was found."

            )





        return answer