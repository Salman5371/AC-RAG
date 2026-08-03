from query_decomposer import QueryDecomposer



decomposer = QueryDecomposer()



question = """

What is Retrieval Augmented Generation?

How does RAG reduce hallucination?

Explain the role of quantum computing in improving RAG retrieval systems.

"""



result = decomposer.decompose(

    question

)



print("\n========== RESULT ==========")


for q in result:

    print(q)