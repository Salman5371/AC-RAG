import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


# Model name
model_name = "Qwen/Qwen2.5-3B-Instruct"


# Check GPU
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Load tokenizer
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# 4-bit Quantization configuration
print("\nPreparing 4-bit configuration...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    # Compute using FP16
    bnb_4bit_compute_dtype=torch.float16,

    # Quantization method
    bnb_4bit_quant_type="nf4",

    # Extra memory optimization
    bnb_4bit_use_double_quant=True
)


# Load model
print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=torch.float16
)


print("\nModel loaded successfully!")


# Test question
question = """
Explain Retrieval Augmented Generation (RAG)
and why it is useful in Generative AI.
"""


# Tokenize input

inputs = tokenizer(
    question,
    return_tensors="pt"
).to(model.device)


# Generate answer

print("\nGenerating answer...")

with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=250,

        temperature=0.7,

        do_sample=True,

        pad_token_id=tokenizer.eos_token_id
    )


# Decode

answer = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)


print("\n==============================")
print("ANSWER:")
print("==============================")

print(answer)