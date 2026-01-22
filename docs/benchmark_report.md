# Comparative Benchmark Report

## Methodology
- **Ground Truth**: Strict labels (2 Good, 3 Partial, 5 Poor) based on updated JD "Required" vs "Preferred" sections. Test files excluded.
- **Data**: 10 resumes (fair distribution).
- **Environment**: Same machine, same data files, robust error handling.

## Results Summary

| Approach | nDCG@3 | Precision@1 | Recall@3 | Speed | Cost |
|----------|--------|-------------|----------|-------|------|
| **V1 (Pure LLM)** | 0.717 | 0.000 | 0.500 | Slow (~13s/resume) | High (2 calls) |
| **V2 (Deterministic)** | **0.965** 🏆 | **1.000** | **1.000** | **Fast (0.03s)** | **Zero** |
| **V3 (Hybrid)** | 0.867 | 0.000 | 0.500 | Medium (5-9s) | Low (1 call) |

## Analysis

### Why V2 (Deterministic) Won?
1. **Semantic Alignment**: The vector embeddings (HuggingFace `all-MiniLM-L6-v2`) provided extremely high fidelity matching to the "Required" section of the JD.
2. **Strictness**: The deterministic approach is unforgiving of missing keywords/skills, which aligned perfectly with the "Strict" ground truth labels (Mike/Sam = 0.0).
3. **No Hallucination**: Unlike V1/V3 LLMs which sometimes "hallucinate" relevance or give "benefit of the doubt" (scoring 0.4 instead of 0.0), V2 correctly assigned low scores to poor candidates.

### Why V1 (LLM) Struggled?
1. **Over-Evaluation**: The LLM often reasoned that candidates like Sarah (Support Lead) had "transferable skills" and gave high scores (0.8+), whereas the strict ground truth penalized her for missing GenAI experience.
2. **Prompts vs Embeddings**: Even with "Critical" instructions, the LLM's inherent bias to be helpful/positive led to score inflation for partial matches.

## Recommendation
- **For Accuracy**: Use **V2 (Deterministic)**.
- **For Explainability**: Use **V1 (LLM)** (provides reasoning).
- **Production Choice**: **V3 (Hybrid)** offers a balance, but requires tuning to prevent LLM score inflation.
