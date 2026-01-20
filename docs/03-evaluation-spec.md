# Resume Matching Engine - Evaluation Specification

> **Version**: 1.0  
> **Status**: DRAFT - Pending Review  
> **Last Updated**: 2026-01-19

---

## 1. Overview

This document specifies the evaluation methodology for the resume matching engine, including:
- Metrics for measuring system performance
- Evaluation procedures
- Success criteria
- Future scaling considerations

---

## 2. Evaluation Dataset

### 2.1 Current Dataset Composition

| Category | Count | Expected Score |
|----------|-------|----------------|
| Good Match | 3 | 0.75 - 1.00 |
| Partial Match | 4 | 0.40 - 0.74 |
| Poor Match | 3 | 0.00 - 0.39 |
| **Total** | **10** | - |

### 2.2 Dataset Limitations

- Small sample size (6 resumes)
- Single job description
- Limited diversity in roles/industries
- Manual labeling (potential bias)

---

## 3. Core Evaluation Metrics

To demonstrate **AI Evaluation Skills**, we will use a tiered metric approach that addresses both ranking quality and classification purity.

### 3.1 Primary Metric: nDCG @ K (Normalized Discounted Cumulative Gain)
**Why?** This is the gold standard for ranking evaluation in search systems. It measures how well the system places highly relevant resumes at the top, penalizing errors at the first position more than the fifth.
- **Target**: nDCG@3 ≥ 0.90
- **Relevance Levels**: 1.0 (Good), 0.5 (Partial), 0.0 (Poor)

### 3.2 Secondary Metrics: Precision & Recall at K
**Why?** These measure the "purity" of the top results, critical for a recruiter's trust. 
- **Precision@1**: Is the top-ranked resume actually a "Good" match? (Target: 100%)
- **Recall@2**: Are both "Good" matches found in the top 2 results? (Target: 100%)

### 3.3 POC Metric: Pairwise Ranking Accuracy
**Definition**: Percentage of resume pairs where the system correctly ranks the better candidate higher.
- **Target**: ≥ 90% accuracy
- **Formula**: `(Correct Pairs) / (Total Unique Pairs with different labels)`

---

### 3.2 Secondary Metrics (For Analysis)

#### 3.2.1 Component Score Correlation

Analyze how each scoring component contributes to final accuracy:

```python
def component_correlation(results: list[dict]) -> dict:
    """
    Pearson correlation between each component score and ground truth.
    """
    ground_truth = [r["label_score"] for r in results]
    
    return {
        "semantic_correlation": pearsonr(
            [r["semantic_score"] for r in results], ground_truth
        )[0],
        "cross_encoder_correlation": pearsonr(
            [r["cross_encoder_score"] for r in results], ground_truth
        )[0],
        "skill_match_correlation": pearsonr(
            [r["skill_match_score"] for r in results], ground_truth
        )[0],
        "final_score_correlation": pearsonr(
            [r["final_score"] for r in results], ground_truth
        )[0],
    }
```

**Purpose**: Identify which component is most predictive

---

#### 3.2.2 Skill Extraction Accuracy

```python
def skill_extraction_evaluation(results: list[dict]) -> dict:
    """
    Evaluate skill extraction quality.
    """
    total_expected = 0
    total_found = 0
    total_correct = 0
    
    for r in results:
        expected = set(r["expected_skills"])
        found = set(r["extracted_skills"])
        
        total_expected += len(expected)
        total_found += len(found)
        total_correct += len(expected & found)
    
    precision = total_correct / total_found if total_found > 0 else 0
    recall = total_correct / total_expected if total_expected > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

---

### 3.3 Future Metrics (Larger Dataset)

When a labeled production dataset is available, we would measure:

#### 3.3.1 Normalized Discounted Cumulative Gain (nDCG)

**Definition**: Measures ranking quality with graded relevance.

```python
def ndcg_score(y_true: list[float], y_pred: list[float], k: int = None) -> float:
    """
    Calculate nDCG@k for ranking evaluation.
    
    y_true: Ground truth relevance scores
    y_pred: Predicted scores used for ranking
    """
    from sklearn.metrics import ndcg_score as sklearn_ndcg
    return sklearn_ndcg([y_true], [y_pred], k=k)
```

**Why nDCG?**
- Ideal for graded relevance (0.0, 0.5, 1.0)
- Penalizes errors at top of ranking more heavily
- Industry standard for search/IR evaluation

---

#### 3.3.2 Precision@K and Recall@K

```python
def precision_at_k(results: list, k: int, threshold: float = 0.5) -> float:
    """
    Of the top K predicted, how many are actually relevant?
    """
    sorted_results = sorted(results, key=lambda x: x["predicted_score"], reverse=True)
    top_k = sorted_results[:k]
    
    relevant_in_top_k = sum(1 for r in top_k if r["label_score"] >= threshold)
    return relevant_in_top_k / k

def recall_at_k(results: list, k: int, threshold: float = 0.5) -> float:
    """
    Of all relevant results, how many are in top K?
    """
    sorted_results = sorted(results, key=lambda x: x["predicted_score"], reverse=True)
    top_k_ids = {r["resume_id"] for r in sorted_results[:k]}
    
    total_relevant = sum(1 for r in results if r["label_score"] >= threshold)
    relevant_in_top_k = sum(
        1 for r in results 
        if r["label_score"] >= threshold and r["resume_id"] in top_k_ids
    )
    
    return relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
```

---

#### 3.3.3 Mean Reciprocal Rank (MRR)

**Definition**: Average of reciprocal ranks of first relevant result.

```python
def mean_reciprocal_rank(queries_results: list[list[dict]]) -> float:
    """
    MRR across multiple queries (job descriptions).
    """
    reciprocal_ranks = []
    
    for results in queries_results:
        sorted_results = sorted(results, key=lambda x: x["predicted_score"], reverse=True)
        for rank, r in enumerate(sorted_results, 1):
            if r["label"] == "good":
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)
```

---

## 4. Evaluation Procedure

### 4.1 Step-by-Step Protocol

```
1. SETUP
   ├── Load job description from data/job_descriptions/
   ├── Load all resumes from data/resumes/
   └── Load ground truth from data/labeled_dataset.json

2. PREPROCESSING
   ├── Apply text preprocessing to JD and all resumes
   └── Log preprocessing statistics

3. SCORING
   ├── For each resume:
   │   ├── Compute semantic score
   │   ├── Compute cross-encoder score
   │   ├── Compute skill match score
   │   ├── Compute weighted final score
   │   └── Extract matched/missing skills
   └── Store all results

4. EVALUATION
   ├── Calculate pairwise accuracy
   ├── Calculate Coverage@2
   ├── Analyze score distributions
   ├── Calculate component correlations
   └── Generate visualizations

5. REPORTING
   ├── Generate summary table
   ├── Create ranking comparison chart
   ├── Document findings and limitations
   └── Suggest improvements
```

### 4.2 Expected Output Format

```json
{
  "evaluation_summary": {
    "pairwise_accuracy": 0.93,
    "coverage_at_2": 1.0,
    "good_mean_score": 0.82,
    "partial_mean_score": 0.58,
    "poor_mean_score": 0.21
  },
  "per_resume_results": [
    {
      "resume_id": "resume_001",
      "label": "good",
      "label_score": 1.0,
      "predicted_score": 0.85,
      "semantic_score": 0.82,
      "cross_encoder_score": 0.88,
      "skill_match_score": 0.80,
      "matched_skills": ["python", "fastapi", "postgresql", "aws", "docker"],
      "missing_skills": ["kubernetes"]
    }
    // ... more results
  ],
  "ranking": [
    {"rank": 1, "resume_id": "resume_001", "score": 0.85, "label": "good"},
    {"rank": 2, "resume_id": "resume_002", "score": 0.81, "label": "good"},
    // ...
  ]
}
```

---

## 5. Visualization Requirements

### 5.1 Score Distribution Plot

```
                 Score Distribution by Category
         
    Good    ████████████████████████████████░░░░░ (0.75-0.90)
    Partial ░░░░░░░░░░████████████████░░░░░░░░░░░ (0.45-0.65)
    Poor    ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (0.15-0.30)
            |-------|-------|-------|-------|-----|
            0.0     0.25    0.50    0.75    1.0
```

### 5.2 Ranking Comparison Table (Top 5)

| Rank | Resume | Predicted | Actual Label | ✓/✗ |
|------|--------|-----------|--------------|-----|
| 1 | Alex Chen | 0.88 | Good | ✓ |
| 2 | Sarah Johnson | 0.84 | Good | ✓ |
| 3 | Maya Gupta | 0.81 | Good | ✓ |
| 4 | David Kim | 0.58 | Partial | ✓ |
| 5 | Priya Sharma | 0.54 | Partial | ✓ |

### 5.3 Component Contribution Analysis

```
Component Correlation with Ground Truth:

Cross-Encoder:  ████████████████████████░ 0.92
Semantic:       ██████████████████████░░░ 0.85
Skill Match:    ████████████████░░░░░░░░░ 0.71
Final (weighted): ████████████████████████░ 0.94
```

---

## 6. Success Criteria

### 6.1 Must-Have (POC Acceptance)

| Criterion | Target | Status |
|-----------|--------|--------|
| nDCG @ 3 ≥ 0.85 | Required | ⏳ |
| Good average > Partial average > Poor average | Required | ⏳ |
| Pairwise accuracy ≥ 85% | Required | ⏳ |
| Precision@1 = 100% | Required | ⏳ |

### 6.2 Nice-to-Have (Excellence)

| Criterion | Target | Status |
|-----------|--------|--------|
| Pairwise accuracy ≥ 90% | Stretch | ⏳ |
| Coverage@2 = 100% | Stretch | ⏳ |
| Clear score separation (gap > 0.2) | Stretch | ⏳ |
| All component correlations > 0.7 | Stretch | ⏳ |

---

## 4. Demonstrating "AI Evaluation" Skills

In the final report, we will go beyond simple numbers and demonstrate specialized AI evaluation techniques:

### 4.1 LLM Consistency Check
Since LLMs can be stochastic, we will run the same resume-JD pair 3 times and measure the standard deviation of the scores. This proves we care about **Reliability**.

### 4.2 Prompt Sensitivity Analysis
We will experiment with different prompt versions (e.g., changing the weighting of "years of experience") to see how it affects the ranking of marginal candidates. This shows **Prompt Engineering** mastery.

### 4.3 Semantic Audit (Embedding Baseline)
We will use the deterministic Embedding scores to "audit" the LLM. If the LLM gives a high score while the Embedding score is very low, we will flag it for human review or investigation. This demonstrates **System Safety & Guardrails**.

### 4.4 Metric Selection Justification
In the README, we will argue why **nDCG** is the superior metric for recruitment (where find the *best* candidate matters more than finding *all* candidates).
