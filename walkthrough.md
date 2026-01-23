# Walkthrough: Resume Matcher Optimization & Benchmark

## Task Overview
We aimed to improve the accuracy and fairness of the resume matching system (V1) and compare it against deterministic (V2) and hybrid (V3) approaches.

## Key Actions Taken
1. **Refined Job Description**: Split into strict **Required** (Python, GenAI) and **Preferred** sections.
2. **Created Strict Ground Truth**: Manually relabeled 10 resumes (2 Good, 3 Partial, 5 Poor). Removed "halo effect" of experience.
3. **Improved V1 Scorer (LLM)**: Added Per-Experience Relevance Scaling and Formula Keys.
4. **Refined V3 (Hybrid)**: Converted to a **Parallel Ensemble** (Average of V1 + V2) for robust scoring.
5. **Cleaned Codebase**: Centralized config, extensive logging, removed hardcoded values.

## Benchmark Results (Strict Evaluation)

| Approach | nDCG@3 | Precision@1 | Key Insight |
|----------|--------|-------------|-------------|
| **V2 (Deterministic)** | **0.965** | **1.000** | **Winner**: Vector embeddings perfectly captured strict requirements. 0% Hallucination. |
| **V3 (Hybrid Ensemble)**| **~0.93** | 1.000 | **Balanced**: Smooths out V1's leniency with V2's strictness. Safe bet. |
| **V1 (LLM)** | 0.899 | 1.000 | **Improved**: Date bug fix raised score from 0.717, but still overrates partial matches. |

*Note: V1 score improved significantly after fixing a prompt bug where "Present" was interpreted as 2025 instead of 2026.*

## Model Limitations

| Model | Strengths | Limitations |
| :--- | :--- | :--- |
| **V1 (Pure LLM)** | Excellent reasoning, understands nuances (e.g., "Docker" implies "Containerization"). | **Hallucinations**: Can invent skills.<br>**Bias**: Overvalues "Preferred" skills (e.g., Support experience).<br>**Cost/Speed**: Expensive & Slow. |
| **V2 (Deterministic)** | **0% Hallucination**, Fast, Cheap, Strict. | **Rigid**: Misses synonyms not in regex (unless embeddings catch them).<br>**No Reasoning**: Can't explain *why* beyond keywords.<br>**Zero-Shot**: Needs taxonomy maintenance. |
| **V3 (Hybrid Ensemble)** | Balanced score, robust to single-model failure. | **Complexity**: Runs 2 pipelines.<br>**Averaging Effect**: Can dilute a "perfect" V2 rejection with a V1 hallucination (and vice versa). |

## Future Improvements

### 1. Recency Scoring (Decay Function)
Currently, 1 year of experience in 2015 counts the same as 1 year in 2025.
**Proposal:** Apply a decay factor to the relevance score:
`Score = Duration * (1 / (Current_Year - End_Year + 1))`
This would prioritize recent GenAI experience over legacy Java experience, which is critical for this role.

### 2. Retrieve & Rank (RAG)
Instead of scoring everyone with LLM (V3 Ensemble), use V2 to **Filter** the top 20%, then use V1 to **Re-rank** only the survivors. This cuts cost by 80%.

## Final Ranking (Best to Worst)

1. **V2 (Deterministic)**: Most reliable for strict filtering.
2. **V3 (Hybrid)**: Best balance if you need reasoning but want safety.
3. **V1 (LLM)**: Good for "discovery" but too risky for automated filtering.

## Artifacts
- `docs/benchmark_report.md`: Detailed analysis.
- `docs/labeling_rationale.md`: Justification for ground truth labels.
- `runs/`: Full audit trails.
