# 🛠️ Resume Matcher V2: Pure Deterministic Approach

> **Version 2 of 3** — Moving from "Black Box" reasoning to "Auditable Facts"

## The Pivot

V1 taught us a painful lesson: **LLMs struggle with reliability.** 

Even with 70B parameter models and aggressive prompt engineering, V1 suffered from:
- **Inter-stage amnesia:** Parser extracted a skill, Scorer called it "missing".
- **Semantic drift:** Missing simple tool matches (e.g., missed Prometheus as a "logging tool").
- **High cost:** $0.02/resume is too much for batch processing.

**V2 Motto: "If it can be a regex, it should be a regex."**

---

## 🏗️ Architecture

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

## 📊 Results (V2 vs V1)

| Metric | V1 (Pure LLM) | V2 (Deterministic) | Improvement |
|--------|---------------|--------------------|-------------|
| **nDCG@3** | 0.837 | [TBD] | - |
| **Precision@1** | 1.000 | [TBD] | - |
| **Cost / 1k** | ~$20.00 | $0.00 | **100% cheaper** |
| **Latency / Res** | ~3.5s | <0.1s | **35x faster** |

---

## 🔎 Why This is Better

1. **Zero Hallucination:** If the resume says "Python", the system finds "Python". No imaginary skills.
2. **Infinite Auditability:** Every point in the score is traceable to a specific line in the resume text via regex or keyword density.
3. **Hyper-Scale:** You can process 10,000 resumes in seconds on a single CPU.

---

## 📂 New Components

- **`src/deterministic.py`**: The "Fact Extractor". Uses complex regex patterns for dates and a curated skill taxonomy.
- **`src/scorers/semantic.py`**: Uses `all-MiniLM-L6-v2` embeddings to catch the "vibe" of the resume that regex misses.
- **`src/deterministic_engine.py`**: The orchestrator that merges the two signals.

---

## 💡 The Tradeoffs (The "V3" Setup)

While V2 is reliable, it is **rigid**:
- ❌ **No Nuance:** It doesn't know that "3 years of GenAI" is better than "10 years of COBOL".
- ❌ **Taxonomy Bound:** If a skill isn't in our `REQUIRED_SKILLS` list, it doesn't count, even if it's a perfect synonym.
- ❌ **Context Blind:** It counts "Python" even if it says "I want to learn Python".

**The Goal for V3:** Use V2 as the "Ground Truth" foundation, and let the LLM do the high-level reasoning *only* using those validated facts.

---

## 🚀 Run V2

```bash
git checkout v2-deterministic
pip install -r requirements.txt
python evaluate_v2.py
```
