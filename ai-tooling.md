# AI Tooling Usage

## Tools Used

### 1. ChatGPT (Primary)
Used for:
- Designing RAG architecture
- Writing and debugging code
- Generating evaluation scripts
- Creating synthetic policy documents

**What worked well:**
- Rapid prototyping of full pipeline
- Clear explanations of trade-offs
- Helped structure evaluation methodology

**Limitations:**
- Required manual validation of generated code
- Some tuning still required for optimal performance

---

### 2. HuggingFace Models
Used for:
- Embeddings (MiniLM model)

**What worked well:**
- Free and reliable
- Easy integration

**Limitations:**
- Slightly lower accuracy than paid APIs

---

### 3. Groq API
Used for:
- Fast LLM inference

**What worked well:**
- Extremely low latency
- Good instruction following

**Limitations:**
- Limited model selection

---

## Summary

AI tools significantly accelerated development, especially in:
- Code generation
- System design
- Dataset creation

However, human validation and tuning were necessary to ensure correctness and performance.