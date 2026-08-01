from qwen_generator import QwenGenerator



print("Starting Qwen Test...")



generator = QwenGenerator()



context = """
Retrieval Augmented Generation (RAG) 
is a technique that improves Large Language Models
by retrieving external knowledge from databases
and combining it with generated responses.

RAG reduces hallucination and improves factual accuracy.
"""



question = input(
    "\nEnter question: "
)



answer = generator.generate(
    question,
    context
)



print(
    "\n========== FINAL ANSWER =========="
)


print(answer)