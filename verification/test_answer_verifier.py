from answer_verifier import AnswerVerifier



verifier = AnswerVerifier()



answer = """
RAG improves large language models by retrieving external knowledge.
RAG completely removes hallucination from language models.
RAG reduces factual errors by providing relevant context.
"""



documents = [

"""
Retrieval-Augmented Generation enhances LLMs
by retrieving relevant document chunks from external
knowledge bases.
It helps reduce factual inaccuracies and hallucinations.
"""

]



result = verifier.verify(

    answer,

    documents

)


print("\n========== RESULT ==========")

print(result)