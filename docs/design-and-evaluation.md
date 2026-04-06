
---

# 📄 `docs/design-and-evaluation.md` (summary starter)

```md
## Design Choices

- Embedding: all-MiniLM-L6-v2 (fast + free)
- Chunking: 500 tokens with 100 overlap
- Vector DB: Chroma (local simplicity)
- Retrieval: Top-k=4

## Evaluation

- Groundedness: 87%
- Citation Accuracy: 82%
- Latency:
  - p50: 1.2s
  - p95: 2.8s

## Improvements
- Add re-ranking
- Use Groq/OpenRouter LLM