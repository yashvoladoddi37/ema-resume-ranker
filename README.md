# Resume Matcher V2: Deterministic Approach

> **Version 2 of 3** — Transitioning from opaque LLM reasoning to auditable rule-based extraction

## Motivation

V1 revealed fundamental limitations of pure LLM pipelines:
- **Inter-stage context loss:** Parser extracted a skill, Scorer reported it missing
- **Semantic drift:** Failed to match tools like Prometheus as "logging infrastructure"
- **Cost:** $0.02/resume is prohibitive for batch processing

**V2 Principle: "If it can be expressed as a pattern, use a pattern."

---

## Architecture

V2 removes the LLM entirely, replacing it with a hybrid of rule-based extraction and vector similarity.

```
Resume.txt
    │
    ├──────────────────────────────┐
    ▼                              ▼
┌──────────────────────┐  ┌──────────────────────┐
│  REGEX / KEYWORDS    │  │  SEMANTIC EMBEDDINGS │
│  (Deterministic)     │  │  (Sentence-Xfmrs)    │
│                      │  │                      │
│  - Years of Exp      │  │  Cosine Similarity   │
│  - Exact Skill Match │  │  between Resume      │
│  - Domain Relevance  │  │  and Job Description │
└──────────────────────┘  └──────────────────────┘
    │                              │
    └──────────────┬───────────────┘
                   ▼
          Final Score (Weighted)
            • 70% Deterministic
            • 30% Semantic
```

---

## Results (V2 vs V1 - Strict Benchmark)

| Metric | V1 (Pure LLM) | V2 (Deterministic) | Delta |
|--------|---------------|--------------------|-------|
| **nDCG@3** | 0.717 | **0.965** | **+35% (Superior)** |
| **Precision@1** | 0.000 | **1.000** | **Perfect** |
| **Cost / 1k resumes** | ~$20.00 | $0.00 | 100% reduction |
| **Latency / resume** | ~13s | <0.1s | 130x faster |

*Based on strict evaluation excluding test files.*

---

## Advantages

1. **No hallucination:** If the resume states "Python", the system matches "Python". No inferred skills.
2. **Full auditability:** Every score component traces to specific text via regex or keyword density.
3. **Scalability:** Process 10,000 resumes in seconds on a single CPU.

---

## Components

| File | Purpose |
|:-----|:--------|
| `src/deterministic.py` | Rule-based extraction: regex patterns for dates, curated skill taxonomy |
| `src/scorers/semantic.py` | Semantic similarity via `all-MiniLM-L6-v2` embeddings |
| `src/deterministic_engine.py` | Orchestrator combining both signals |

---

## Limitations

V2 is reliable but inflexible:
- **No contextual reasoning:** Cannot distinguish "3 years of GenAI" from "10 years of COBOL"
- **Taxonomy-bound:** Skills outside `REQUIRED_SKILLS` are ignored, even valid synonyms
- **Context-blind:** Matches "Python" even in "I want to learn Python"

**V3 Goal:** Use V2 as the ground truth foundation, with LLM reasoning constrained to validated facts.

### Qualitative Analysis: The Precision of Strictness

V2 eliminates hallucination but introduces a new challenge: **Semantic Brittleness**.

**Case Study 1: Sarah Johnson (Correct Rejection)**
- **V2 Score**: **0.52** (Partial/Low)
- **Why**: Her resume lacked the exact tokens for "GenAI" or "LLM" in the required section.
- **Outcome**: Unlike V1, V2 correctly identified that while she is senior, she is not an *AI* engineer. The zero score in `ai_relevance` accurately reflected the hard constraint.

**Case Study 2: The "Vocabulary Gap" (Maya vs. Variations)**
- **V2 Score**: 0.633 (Maya)
- **Problem**: If a candidate says "built retrieval systems" instead of "RAG", V2's regex might miss it entirely.
- **Outcome**: High precision, lower recall for semantically equivalent but lexically distinct terms.

---

## Usage

```bash
git checkout v2-deterministic
pip install -r requirements.txt
python evaluate_v2.py
```
