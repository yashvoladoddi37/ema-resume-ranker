# Resume Matcher V1: Two-Stage LLM Pipeline

> **Version 1 of 3** — Iterative development of an AI-powered resume matching system

## Hypothesis

**"Delegate all extraction and scoring to the LLM."**

With capable LLMs like Llama 3.3 70B, a two-call pipeline should suffice:
1. **Stage 1 (Parser):** Extract structured data from unstructured resume text
2. **Stage 2 (Scorer):** Evaluate parsed data against job requirements

---

## Architecture

```
Resume.txt
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 1: LLM PARSER                │
│  (ResumeParser)                     │
│                                     │
│  Input:  Raw resume text            │
│  Output: Structured JSON            │
│          {                          │
│            skills: [...],           │
│            experience: [...],       │
│            years: 4.0               │
│          }                          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 2: LLM SCORER                │
│  (ResumeScorer)                     │
│                                     │
│  Input:  Parsed JSON + Job Desc    │
│  Output: Dimension scores           │
│          {                          │
│            skill_match: 0.80,       │
│            experience_depth: 0.90,  │
│            domain_fit: 0.90         │
│          }                          │
└─────────────────────────────────────┘
    │
    ▼
Final Score = Σ(dimension × weight)
  • 50% skill match
  • 30% experience depth
  • 20% domain fit
```

---

## Results

Evaluated on 10 labeled resumes for the "AI Applications Engineer" role.

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **nDCG@3** | 0.899 | Significantly improved after date parsing fix (was 0.717) |
| **Precision@1** | 1.000 | Best candidate (Maya) correctly ranked #1 |
| **Recall@3** | 0.500 | Still misses some nuances compared to V2 |

### Top Ranked Candidates (Strict Ground Truth)

```
#1 | Maya Gupta    — 0.907  ✓ (Ground truth: 1.0)
#2 | Sarah Johnson — 0.843  ✓ (Ground truth: 0.5 - False High)
#3 | Yashpreet V.  — 0.832  ✓ (Ground truth: 1.0)
```

**Cost:** ~$0.02 per resume (2 LLM calls × ~500 tokens each)  
**Latency:** ~3.5 seconds per resume

---

## Findings

### What Worked

1. **Nuanced reasoning** — LLM understood context:
   - "3 years in GenAI support" > "5 years in unrelated backend dev"
   - Detected transferable skills: "Docker experience likely covers containerization needs"

2. **Structured output** — JSON mode ensured consistent scoring format

3. **Decent accuracy** — nDCG@3 of 0.899 is competitive, though slightly below Deterministic V2 (0.965).

### What Failed

#### 1. **Hallucinations / Over-Optimism**

Example: Sarah Johnson (Support Engineer) was ranked highly (0.843) despite lacking required GenAI experience. The LLM inferred "transferable skills" where the strict requirements demanded specific experience.

#### 2. **Prompt Fragility**

A previous bug where "Present" was interpreted as "January 2025" (instead of 2026) caused negative duration calculations for current roles, severely penalizing candidates like Yashpreet. Fixing this improved his score from 0.820 to 0.832 and ranking from #5 to #3.

#### 3. **Expensive** 

- 1,000 resumes = $20
- 10,000 resumes = $200

For batch processing, this doesn't scale.

### Qualitative Analysis: The "Reasoning" Trap

While the LLM provides detailed justifications, it often falls into the trap of being "too helpful," finding transferable skills where strict requirements exist.

**Case Study: Sarah Johnson (False Positive)**
- **Role**: Support Lead (no structural AI experience)
- **V1 Score**: **0.843** (Very High)
- **LLM Reasoning**: *"Experience in API troubleshooting is transferable to AI workflows. Her leadership in support roles indicates potential for senior engineering management."*
- **Reality**: The role requires *immediate* contribution to GenAI pipelines. The LLM's optimism led to a false positive ranking, placing her above candidates with actual relevant coding experience.

---

## Next Steps: Version 2

The V1 hypothesis was: **"LLMs can do it all."**

What I learned: **They can, but they shouldn't (for strict filtering).**

### Problems to Solve in V2

1. **Hallucinations** → Need verifiable ground truth
2. **Cost** → Need cheaper baseline
3. **Auditability** → Need explainable scoring components

**Next:** [V2: Deterministic Baseline](../v2-deterministic) →
