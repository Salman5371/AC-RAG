
# AC-RAG
## Adaptive Self-Correcting Retrieval-Augmented Generation System

AC-RAG is a research-oriented Retrieval-Augmented Generation (RAG) framework designed to improve Large Language Model (LLM) reliability by combining hybrid retrieval, reranking, adaptive retrieval strategies, and grounded generation.

The system addresses major LLM limitations such as hallucination, outdated knowledge, and lack of domain-specific information by retrieving relevant external knowledge and generating evidence-based responses.

---

## 🚀 System Architecture

```

User Query
|
↓
Query Complexity Analysis
|
↓
Hybrid Retrieval
(FAISS + BM25)
|
↓
Candidate Documents
|
↓
Cross Encoder Reranking
|
↓
Adaptive Evidence Selection
|
↓
Qwen2.5 Generation
|
↓
Answer + Source Evidence

```

---

# ✨ Key Features

### 🔹 Hybrid Retrieval
Combines:

- **FAISS Vector Search**
  - Semantic understanding using embeddings
  - Handles paraphrased queries

- **BM25 Keyword Search**
  - Captures exact technical terms
  - Improves keyword-based retrieval


### 🔹 Semantic Embedding

Embedding Model:

```

BAAI/bge-small-en-v1.5

```

Vector database:

```

FAISS

```


### 🔹 Cross Encoder Reranking

Model:

```

BAAI/bge-reranker-base

```

The reranker evaluates query-document relevance and prioritizes the most useful evidence.


### 🔹 Adaptive Retrieval

AC-RAG dynamically analyzes query complexity.

Simple query:

```

What is Retrieval Augmented Generation?

```

→ Retrieves focused evidence


Complex query:

```

What methods improve retrieval quality in RAG systems?

```

→ Retrieves broader evidence coverage


### 🔹 LLM Generation

Final answers are generated using:

```

Qwen2.5

```

The model receives retrieved evidence and produces grounded responses.

---

# 📂 Project Structure

```

AC-RAG/

├── retrieval/
│   ├── retriever.py
│   ├── reranker.py
│   ├── adaptive_retriever.py
│   └── test_adaptive.py
│
├── generation/
│   ├── qwen_generator.py
│   └── rag_answer.py
│
├── evaluation/
│   ├── test_questions.json
│   ├── retrieval_metrics.py
│   └── generation_metrics.py
│
├── embeddings/
│   ├── faiss.index
│   └── chunks.pkl
│
├── models/
├── data/
├── requirements.txt
└── README.md

````

---

# ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/yourusername/AC-RAG.git

cd AC-RAG
````

Create environment:

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run System

### Test Adaptive Retrieval

```bash
python retrieval/test_adaptive.py
```

Example:

```
Enter question:
What is Retrieval Augmented Generation?
```

Output includes:

* Retrieved documents
* FAISS score
* Hybrid score
* Reranker score

---

### Generate Final Answer

```bash
python generation/rag_answer.py
```

Output:

```
========== FINAL ANSWER ==========

Generated response...


========== SOURCES ==========

Chunk ID
Reranker Score
Evidence Context
```

---

# 📊 Current Evaluation

## Retrieval Performance

```
Precision@K : 0.617

Hit Rate    : 0.8
```

## Generation Performance

```
Answer Coverage Score : 0.667
```

---

# 🛠️ Technologies

| Component    | Technology        |
| ------------ | ----------------- |
| Language     | Python            |
| Retrieval    | FAISS + BM25      |
| Embedding    | BAAI BGE          |
| Reranking    | BGE Cross Encoder |
| LLM          | Qwen2.5           |
| Architecture | RAG               |

---

# 🔬 Research Contribution

AC-RAG introduces:

* Hybrid semantic + keyword retrieval
* Adaptive query complexity analysis
* Cross-encoder based evidence ranking
* Source-aware answer generation

The framework improves retrieval reliability and reduces irrelevant context before LLM generation.

---

# 🔮 Future Improvements

* RAGAS evaluation
* Semantic answer evaluation
* Metadata-aware retrieval
* Query expansion
* Better chunk optimization
* Hallucination detection

---

## License

Research and educational use.


## Author : Md Salman Farshi.