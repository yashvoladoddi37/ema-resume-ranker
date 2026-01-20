# AI-Powered Resume Matcher: Ema AI Applications Engineer Case Study

> **Assignment Completion**: AI-powered resume matching system demonstrating prompt engineering and AI evaluation skills for the Ema AI Applications Engineer role.

## 📌 Executive Summary

This project implements a production-grade resume matching engine specifically tailored to the **Ema AI Applications Engineer** role. It ranks candidates against the official job description using an **LLM-based scoring system** powered by Groq's `llama-3.3-70b-versatile` model.

The solution demonstrates:
1. **Prompt Engineering**: Structured, reasoning-heavy LLM prompts with JSON schema enforcement
2. **AI Evaluation**: Objective assessment with detailed technical justifications
3. **Explainability**: Human-readable reasoning and skill gap analysis for every candidate

---

### 1. Model & Algorithm Selection: Sequential Hybrid Orchestration
For the core matching task, I implemented a **Sequential Hybrid Engine** (60% LLM + 40% Deterministic).

**The Workflow:**
1. **Deterministic Stage (Ground Truth)**: Python extracts verifiable facts (years, skills, keyword density).
2. **Context Enrichment**: These facts are injected into the LLM prompt as "System Detected Facts."
3. **LLM Stage (Cognitive Layer)**: The LLM (Llama 3.3 70B) evaluates the *quality* and *relevance* of the resume using the deterministic facts as an anchor.
4. **Weighted Aggregation**: Final score combines both signals.

### ⚖️ Justification & Alternatives
#### Why this augmented approach?
It solves the "Hallucination-Consistency" problem. By feeding deterministic facts into the LLM, we ensure the LLM reasoning doesn't contradict the data (e.g., claiming a candidate has 5 years when the system only found 2).

| Approach | Advantages | Disadvantages |
|-----------|------------|---------------|
| **Parallel Hybrid** | Fast, independent | LLM reasoning may contradict hard rules |
| **Sequential (Chosen)** | **High Consistency + Reasoning** | Slightly higher prompt tokens |
| **Pure Semantic** | Simple baseline | Fails on "hard bars" (e.g. mandatory skills) |

**Alternative Considered**: I considered a **Pure Semantic Embedding (Cosine Similarity)** approach. While faster, I rejected it as the primary engine because embedding similarity often ignores "hard bars" (e.g., a candidate scoring high on similarity due to a generic stack but missing the mandatory 3+ years of experience).

### 🧪 Feature Engineering & Preprocessing
To clean and prepare the text data:
1. **Cleaning**: Stripping non-essential characters, normalizing case, and removing stop-words for deterministic matching.
2. **Entity Extraction**: Using regex anchors to extract numeric years of experience and specific technical keywords (e.g., mapping "GCP" and "Google Cloud" to the same skill bucket).
3. **Structured Context**: Feeding the LLM cleaned snippets rather than raw, noisy text to reduce token usage and improve reasoning focus.

---

## 📊 Evaluation Results

### Dataset
**10 synthetic resumes** representing three tiers:
- **Good Match (2)**: Strong GenAI/Agentic experience + Enterprise SaaS Support
- **Partial Match (5)**: Technical gaps in either AI workflows or Customer Success
- **Poor Match (3)**: Irrelevant stack (Frontend, Java, Marketing)

### Rankings (Hybrid Scoring)

| Rank | Candidate | Final | LLM | Deterministic | YRS | Assessment |
|------|-----------|-------|-----|---------------|-----|------------|
| 1 | Maya Gupta | 0.787 | 0.90 | 0.618 | 4 | ✅ Correct |
| 2 | Sarah Johnson | 0.783 | 0.90 | 0.608 | 7 | ✅ Correct |
| 3 | David Kim | 0.595 | 0.70 | 0.437 | 5 | ✅ Correct |
| 4 | Yashpreet | 0.544 | 0.70 | 0.309 | 0* | ✅ Correct |
| 5 | Priya Sharma | 0.492 | 0.60 | 0.331 | 5 | ✅ Correct |
| 6 | Mike | 0.356 | 0.40 | 0.290 | 4 | ✅ Correct |
| 7 | Sam | 0.322 | 0.30 | 0.356 | 6 | ✅ Correct |
| 8-10 | James/Jordan/Elena | 0.08-0.20 | 0.00-0.20 | 0.20-0.29 | 3-8 | ✅ Correct |

\* *Yashpreet's resume didn't explicitly state years (deterministic extracted 0), but LLM inferred seniority from context.*

### Key Metrics
- **Top-2 Accuracy**: 100% (best "Good" candidates in top 2)
- **Tier Separation**: Clear gaps - Good (0.78+), Partial (0.32-0.60), Poor (<0.20)
- **Deterministic Anchoring**: Sarah's 7 years + 87.5% required skill coverage validates her #2 ranking
- **JSON Success Rate**: 100% (all 10 evaluations returned valid structured output)

---

## 💡 Prompt Engineering Highlights

### Structured Prompt Design
```python
You are a Principal AI Applications Engineer at Ema. Evaluate this candidate...

### Instructions:
1. Analyze with Nuance: Look beyond keyword matching
2. Score (0.0 - 1.0): [detailed rubric]
3. Reasoning: Provide 2-3 sentence technical justification

### Output Format:
{
  "score": float,
  "reasoning": "string",
  "matched_skills": ["list"],
  "missing_skills": ["list"]
}
```

### Techniques Used
- **Role Prompting**: Setting persona as "Principal Engineer" for domain expertise
- **Structured Output**: JSON schema enforcement via `response_format`
- **Explicit Rubric**: Clear 0.0/0.5/1.0 scoring guidelines
- **Chain-of-Thought**: Instructing model to provide reasoning before scoring

---

## 🚀 Setup & Reproducibility

### Prerequisites
- Python 3.9+
- Groq API Key (or Gemini as fallback)

### Installation
```bash
git clone <your-repo>
cd ema
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-minimal.txt
```

### Configuration
Create `.env` file:
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
LLM_MODEL=llama-3.3-70b-versatile
```

### Running the Demo
```bash
# Hybrid Engine (LLM + Deterministic)
python demo_hybrid.py

# LLM-Only (minimal dependencies)
python demo_minimal.py
```

**Output**: 
- Console table with rankings
- `results_llm_only.json` with detailed scores and reasoning

---

## 📁 Project Structure

```
ema/
├── data/
│   ├── job_descriptions/
│   │   └── ema_ai_apps_engineer.txt
│   ├── resumes/
│   │   ├── res_001_yashpreet.txt
│   │   └── ... (10 total)
│   └── labeled_dataset.json
├── src/
│   ├── config.py
│   ├── prompts.py
│   ├── preprocessing.py
│   ├── engine.py
│   └── scorers/
│       ├── llm.py
│       └── semantic.py
├── docs/
│   ├── 01-architecture-spec.md
│   ├── 02-data-spec.md
│   ├── 03-evaluation-spec.md
│   └── 04-final-readme-draft.md
├── demo_minimal.py
├── requirements-minimal.txt
├── EVALUATION_RESULTS.md
└── README.md
```

---

## 🔍 Sample LLM Reasoning

**Top Candidate - Maya Gupta (0.85)**:
> "The candidate's experience in configuring and integrating AI/GenAI workflows, troubleshooting production issues, and proficiency in Python and APIs aligns well with the job requirements. Additionally, their expertise in prompt engineering and LangChain is a strong asset."

**Objective Assessment - Yashpreet (0.80)**:
> "The candidate has relevant experience in AI engineering, proficiency in Python, and familiarity with APIs and integration tools. However, the candidate's experience in technical support or customer success for SaaS is limited... The candidate's skills in Golang, LangGraph, and LLM integrations are a plus."

---

## 🎯 Future Improvements & Measurement

### Formal Success Metrics
With a larger, labeled dataset, I would measure success using:
1. **nDCG (Normalized Discounted Cumulative Gain)**: To evaluate the quality of the *ranking order*, which is crucial for recruiters who only look at the top 10.
2. **Mean Reciprocal Rank (MRR)**: To ensure the "perfect match" is ideally at Rank #1.
3. **Precision@K**: To measure how many "Good Matches" are surfaced in the first `k` results.

### 🛠️ Technical Discovery: The "Semantic Gap"
During testing, we discovered a key challenge: **Deterministic Brittleness**. 
- **The Issue**: The rule-based extractor missed "REST" because the candidate wrote "RESTful". It missed "Years of Experience" because it looked for "X years" while the candidate used date ranges ("2024-2025").
- **The Hybrid Fix**: 
    - The **Deterministic Layer** provides the rigid, auditable base.
    - The **LLM Layer** acts as the "Intelligent Buffer," correctly identifying the seniority and skills that the rigid rules missed.
- **Lesson**: Relying *only* on deterministic rules leads to high False Negatives. Relying *only* on LLM leads to high variance. **Hybrid is the production gold standard.**

---

## 📝 Documentation

- **Architecture**: [`docs/01-architecture-spec.md`](docs/01-architecture-spec.md)
- **Data Specification**: [`docs/02-data-spec.md`](docs/02-data-spec.md)
- **Evaluation Metrics**: [`docs/03-evaluation-spec.md`](docs/03-evaluation-spec.md)
- **Results Analysis**: [`EVALUATION_RESULTS.md`](EVALUATION_RESULTS.md)

---

## ✅ Assignment Deliverables

- ✅ **AI-powered matching system** with relevance scoring
- ✅ **Technical justification** (prompt engineering, LLM selection)
- ✅ **Performance evaluation** (rankings, tier separation, JSON success rate)
- ✅ **Well-organized code** (modular structure, clear separation of concerns)
- ✅ **Reproducibility** (requirements, .env template, demo script)

---

**Built by**: Yashpreet Voladoddi  
**Role**: AI Applications Engineer @ Ema (Take-Home Assignment)  
**Date**: January 2026
