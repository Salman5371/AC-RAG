from context_checker import ContextQualityChecker



checker = ContextQualityChecker()



query = "What is Retrieval Augmented Generation?"



documents = [

"Retrieval Augmented Generation improves LLMs by retrieving external knowledge.",

"RAG reduces hallucination by providing relevant context."

]



result = checker.check_context(
    query,
    documents
)


print("\n========== RESULT ==========")

print(result)