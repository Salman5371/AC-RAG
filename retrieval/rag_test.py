import torch

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.huggingface import HuggingFaceLLM


# =========================
# Embedding Model
# =========================

print("Loading embedding model...")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# =========================
# Local Qwen LLM
# =========================

print("Loading Qwen model...")


llm = HuggingFaceLLM(

    model_name="Qwen/Qwen2.5-3B-Instruct",

    tokenizer_name="Qwen/Qwen2.5-3B-Instruct",

    context_window=4096,

    max_new_tokens=256,

    generate_kwargs={
        "temperature":0.7,
        "do_sample":True
    },

    device_map="auto",

    model_kwargs={
        "torch_dtype":torch.float16
    }
)


Settings.llm = llm


# =========================
# Load PDF
# =========================

print("Loading PDF...")


documents = SimpleDirectoryReader(
    "documents"
).load_data()


print(
    "Documents loaded:",
    len(documents)
)



# =========================
# Create Index
# =========================

print("Creating vector database...")


index = VectorStoreIndex.from_documents(
    documents
)



# =========================
# Query Engine
# =========================

query_engine = index.as_query_engine(
    similarity_top_k=3
)



question = input(
    "\nAsk question about PDF: "
)


response = query_engine.query(
    question
)


print("\n================")
print("ANSWER")
print("================")

print(response)