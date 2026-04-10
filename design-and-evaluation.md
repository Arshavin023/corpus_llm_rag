# Design and Evaluation Document

## 1. System Design & Architecture

### Overview
This project implements a Retrieval-Augmented Generation (RAG) system for answering company policy questions. The system retrieves relevant policy documents and uses a Large Language Model (LLM) to generate grounded answers with citations.

---

## Architecture

User Query → Vector Search (Chroma) → Top-K Documents → Prompt Construction → LLM (Groq) → Answer + Citations

---

## Design Decisions

### 1. Embedding Model
**Chosen:** sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)

**Why:**
- Free and fast
- Good semantic similarity performance
- No API cost

---

### 2. Vector Database
**Chosen:** Chroma (local)

**Why:**
- Lightweight and easy to use
- No external dependencies
- Supports persistence

---

### 3. Chunking Strategy
- Chunk Size: 400
- Overlap: 100

**Why:**
- Model performance drop to 86.6% when chunks below and above 400 were used.
- Balances context completeness and retrieval precision
- Overlap prevents loss of important boundary information

---

### 4. Retrieval Strategy
- Top-K = 5

**Why:**
- Small corpus → lower K reduces noise
- Maintains relevant context without overwhelming LLM

---

### 5. LLM Choice
**Chosen:** Groq (llama-3.1-8b-instant)

**Why:**
- Very fast inference (low latency)
- Free tier available
- Good instruction-following capability

---

### 6. Prompt Design
- Strict grounding instruction:
  > "Answer ONLY from context"

- Guardrails:
  - Refuse out-of-scope questions
  - Keep answers concise
  - Always rely on retrieved context

---

### 7. Data Sources
- Mixed formats:
  - Markdown
  - HTML
  - XML
  - TXT

**Why:**
- Demonstrates real-world ingestion capability

---

## 2. Evaluation Approach

### Dataset
- 15–20 manually created evaluation questions
- Covers:
  - PTO
  - Security
  - Expenses
  - Attendance
  - Customer service

---

### Metrics

#### 1. Groundedness (Required)
Definition:
- % of answers fully supported by retrieved context

Implementation:
- Checked via keyword match with expected answer

---

#### 2. Citation Accuracy (Required)
Definition:
- % of answers with correct supporting sources

Implementation:
- Verified if returned citations match expected source

---

#### 3. Latency (Required)
Measured:
- p50 (median)
- p95 latency

---

## 3. Results (Example)

| Metric | Value |
|------|------|
| Groundedness | 0.85 |
| Citation Accuracy | 0.80 |
| Latency p50 | 0.9s |
| Latency p95 | 1.5s |

---

## 4. Observations

- Smaller chunk sizes improved citation accuracy
- Increasing Top-K improved recall but introduced noise
- Overlap improved answer completeness

---

## 5. Future Improvements

- Add re-ranking for better retrieval precision
- Use semantic similarity for evaluation instead of keyword match
- Improve prompt with structured citations
- Add hybrid search (keyword + vector)

---

## 6. Conclusion

The system demonstrates a working RAG pipeline with strong grounding and acceptable latency. Trade-offs between retrieval size and answer quality were observed and optimized.