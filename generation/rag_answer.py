import torch
import faiss
import pickle

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


# ==========================
# Load Vector Database
# ==========================

print("Loading knowledge base...")


index = faiss.read_index(
    "embeddings/faiss.index"
)


with open(
    "embeddings/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)



# ==========================
# Embedding Model
# ==========================

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)



# ==========================
# Reranker
# ==========================

print("Loading reranker...")


reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)



# ==========================
# Qwen Model
# ==========================

print("Loading Qwen...")


model_name = "Qwen/Qwen2.5-3B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)


print("System ready!")



# ==========================
# User Question
# ==========================

question = input(
    "\nAsk question: "
)



# ==========================
# Retrieval
# ==========================

query_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True
)


scores, indices = index.search(
    query_embedding,
    20
)


candidates = [
    chunks[i]
    for i in indices[0]
]



# ==========================
# Reranking
# ==========================

pairs = [
    [question, text]
    for text in candidates
]


rerank_scores = reranker.predict(
    pairs
)


ranked = sorted(
    zip(candidates, rerank_scores),
    key=lambda x:x[1],
    reverse=True
)



# Select best evidence

context = "\n\n".join(
    [
        item[0]
        for item in ranked[:3]
    ]
)



# ==========================
# Prompt
# ==========================

prompt = f"""
You are a helpful research assistant.

Answer the question using ONLY the provided context.

Context:
{context}


Question:
{question}


Answer:
"""


# ==========================
# Generate Answer
# ==========================

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)



with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True
    )



answer = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)


print("\n======================")
print("FINAL ANSWER")
print("======================")

print(answer)