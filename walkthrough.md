# Walkthrough: Resume Matcher Optimization & Benchmark

## Task Overview
We aimed to improve the accuracy and fairness of the resume matching system (V1) and compare it against deterministic (V2) and hybrid (V3) approaches.

## Key Actions Taken

1. **Refined Job Description**:
   - Split into strict **Required** (Python, GenAI, APIs) and **Preferred** (Support, Logging) sections.
   - This provided a clearer signal for evaluation algorithms.

2. **Created Strict Ground Truth**:
   - Manually relabeled 10 resumes based on the updated JD.
   - Enforced a balanced distribution: 2 Good, 3 Partial, 5 Poor.
   - **Crucial Change**: Candidates missing "Required" skills (like GenAI) were labeled as Poor/Partial, removing the "halo effect" of experience years.

3. **Improved V1 Scorer (LLM)**:
   - Added **Per-Experience Relevance Scaling**: Calculating relevant years instead of total years.
   - Added **Semantic Skill Matching**: Explicit instructions to map tools (e.g., "LangChain") to requirements ("GenAI Workflows").
   - Added **Formula Keys**: Making the scoring logic auditable.

4. **Conducted Fair Benchmark**:
   - Ran V1, V2, and V3 on the **exact same** ground truth and data.
   - Handled rate limits and dependencies robustly.

## Benchmark Results (Strict Evaluation)

| Approach | nDCG@3 | Precision@1 | Key Insight |
|----------|--------|-------------|-------------|
| **V1 (LLM)** | 0.899 | 1.000 | **Improved**: Fixing a date parsing bug improved ranking, but still slightly below V2. |
| **V2 (Deterministic)** | **0.965** | **1.000** | **Winner**: Vector embeddings perfectly captured the strict semantic requirements without hallucinating. |
| **V3 (Hybrid)** | 0.867 | 0.000 | **Mixed**: Improved on baseline V1, but struggled with strict ranking nuances. |

*Note: V1 score jumped from 0.717 to 0.899 after fixing a prompt bug where "Present" was interpreted as 2025 instead of 2026, causing negative duration for recent roles.*

## Conclusion
The **V2 Deterministic approach is the superior solution** for this specific use case. It correctly penalizes candidates who lack required skills, whereas the LLM tends to be too lenient/imaginative. The "perfect" V1 score (1.0) observed earlier was identified as data leakage (fitting GT to model), which we corrected.

## Artifacts
- `docs/benchmark_report.md`: Detailed analysis.
- `docs/labeling_rationale.md`: Justification for ground truth labels.
- `results_v1_final.json`: V1 outputs.
- `runs/`: Full audit trails for all experiments.
