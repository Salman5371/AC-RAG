import os
import re
import pickle
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==========================
# PDF Loading
# ==========================

pdf_path = "documents/rag_paper.pdf"


print("Reading PDF...")


reader = PdfReader(pdf_path)


full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"



print("Total characters:", len(full_text))



# ==========================
# Remove References
# ==========================

print("Removing references section...")


patterns = [
    r"\nReferences\n",
    r"\nREFERENCES\n",
    r"\nReferences\s"
]


for pattern in patterns:

    match = re.search(
        pattern,
        full_text
    )

    if match:

        full_text = full_text[:match.start()]

        break



print(
    "After cleaning:",
    len(full_text)
)



# ==========================
# Sentence Chunking
# ==========================

print("Creating chunks...")


sentences = re.split(
    r'(?<=[.!?])\s+',
    full_text
)



chunks = []

current_chunk = ""

chunk_size = 800


for sentence in sentences:

    if len(current_chunk) + len(sentence) <= chunk_size:

        current_chunk += " " + sentence

    else:

        chunks.append(
            current_chunk.strip()
        )

        current_chunk = sentence



if current_chunk:

    chunks.append(
        current_chunk.strip()
    )



print(
    "Total chunks:",
    len(chunks)
)



# ==========================
# Embedding
# ==========================

print("Loading embedding model...")


model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)



embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
    show_progress_bar=True
)



# ==========================
# FAISS
# ==========================


dimension = embeddings.shape[1]


index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embeddings
)


print(
    "FAISS created"
)



# ==========================
# Save
# ==========================


os.makedirs(
    "embeddings_v2",
    exist_ok=True
)



faiss.write_index(
    index,
    "embeddings_v2/faiss.index"
)



with open(
    "embeddings_v2/chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )



print("Saved successfully!")