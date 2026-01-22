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

Evaluated on 12 labeled resumes for the "AI Applications Engineer" role.

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **nDCG@3** | 0.837 | Top 3 ranking quality: Good |
| **Precision@1** | 1.000 | Best candidate was actually #1: Perfect |
| **Recall@3** | 0.667 | Caught 2/3 qualified candidates in top 3 |

### Top Ranked Candidates

```
#1 | Maya Gupta    — 0.850  ✓ (Ground truth: top candidate)
#2 | TEST_A        — 0.850
#3 | Sarah Johnson — 0.810  ✓ (Ground truth: qualified)
```

**Cost:** ~$0.02 per resume (2 LLM calls × ~500 tokens each)  
**Latency:** ~2.5 seconds per resume

---

## Quick Start

```bash
# Setup
git clone https://github.com/yashvoladoddi37/ema-resume-ranker.git
cd ema-resume-ranker
git checkout dev  # This is V1
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
echo "GROQ_API_KEY=your_key_here" > .env

# Run evaluation
python evaluate_with_logging.py

# View results
ls runs/run_*/  # Full audit trail of LLM inputs/outputs

# Launch dashboard
streamlit run app.py
```

---

## Findings

### What Worked

1. **Nuanced reasoning** — LLM understood context:
   - "3 years in GenAI support" > "5 years in unrelated backend dev"
   - Detected transferable skills: "Docker experience likely covers containerization needs"

2. **Structured output** — JSON mode ensured consistent scoring format

3. **Decent accuracy** — nDCG@3 of 0.837 is solid for a first approach

### What Failed

#### 1. **Hallucinations**

Example from `res_004_david`:
```json
"matched_skills": ["Python", "REST", "SOAP", "Prometheus", "Kibana"]
```

But the resume only mentioned Prometheus/Kibana in the context of "built monitoring alerts" — the LLM **inferred** proficiency that wasn't explicitly stated. In hiring, you can't afford this.

#### 2. **Not Auditable**

When a candidate asks "why did I score 0.6 on skill_match?", the answer is:
> "The LLM said so."

Not good enough for production.

#### 3. **Expensive**

- 12 resumes = 24 LLM calls = $0.24
- 1,000 resumes = $20
- 10,000 resumes = $200

For batch processing, this doesn't scale.

#### 4. **Latency**

~2.5s per resume means:
- 1,000 resumes = 42 minutes (sequential processing)
- Even with parallelization (10 concurrent), still 4+ minutes

---

## Project Structure

```
ema-resume-ranker/
├── src/
│   ├── resume_parser.py     # Stage 1: LLM extraction
│   ├── resume_scorer.py     # Stage 2: LLM scoring
│   ├── matching_engine.py   # Orchestrator
│   └── utils.py             # Metrics (nDCG, P@k, R@k)
├── data/
│   ├── resumes/             # 12 test resumes
│   ├── job_descriptions/    # Target JD
│   └── ground_truth.json    # Manual labels
├── runs/                    # Audit trails (LLM I/O logs)
├── app.py                   # Streamlit dashboard
├── evaluate.py              # Basic evaluation
└── evaluate_with_logging.py # Full audit trail
```

---

## Audit Trail

Every run saves complete LLM I/O for debugging:

```bash
python evaluate_with_logging.py
```

Creates:
```
runs/run_20260121_131947/
├── 01_raw_resumes/          # Original text
├── 02_parser_prompts/       # What we sent to Stage 1
├── 03_parsed_data/          # What Stage 1 returned
├── 04_scorer_prompts/       # What we sent to Stage 2
├── 05_scorer_outputs/       # What Stage 2 returned
├── 06_final_results/        # Per-candidate final scores
├── all_results.json         # Ranked output
└── metrics.json             # nDCG, P@k, R@k
```

This transparency is crucial for debugging, but it exposes the core issue: **the LLM is a black box**. You can see what it returned, but not *why*.

---

## Next Steps: Version 2

The V1 hypothesis was: **"LLMs can do it all."**

What I learned: **They can, but they shouldn't.**

### Problems to Solve in V2

1. **Hallucinations** → Need verifiable ground truth
2. **Cost** → Need cheaper baseline
3. **Auditability** → Need explainable scoring components

### The V2 Approach: Pure Deterministic

> "What if we remove the LLM entirely and build a fully auditable, cheap baseline?"

**V2 will use:**
- **Embeddings:** Semantic similarity (sentence-transformers)
- **Regex:** Extract years of experience, education
- **Keyword matching:** Skills against a fixed taxonomy

**Expected tradeoffs:**
- Fast, cheap, fully reproducible
- Trade-off: inflexible, misses synonyms, lacks contextual understanding

---

## Technical Details

### LLM Configuration

```python
Model: llama-3.3-70b-versatile (Groq)
Temperature: 0  # Deterministic outputs
Response format: JSON mode
```

### Scoring Weights

```python
weights = {
    'skill_match': 0.50,      # Most important
    'experience_depth': 0.30, # Years + relevance
    'domain_fit': 0.20        # Background alignment
}
```

Tunable via Streamlit dashboard.

### Evaluation Methodology

We treat this as an **Information Retrieval** problem, not classification:
- **nDCG@3:** Are the best candidates ranked highest?
- **Precision@1:** Is the #1 candidate actually qualified?
- **Recall@3:** Did we catch all qualified candidates in top 3?

---

## Reflections

**What I'd do differently:**
1. Start with a deterministic baseline (not LLM)
2. Build hybrid only after understanding where each approach fails
3. Collect more labeled data (12 resumes is too small)

**What I'd keep:**
1. Two-stage separation (parsing vs scoring)
2. Structured JSON outputs
3. Full audit trail logging
4. IR metrics (not accuracy)

---

## Full Results

See `runs/run_20260121_131947/all_results.json` for complete output.

**Distribution:**
- Top performer: 0.850 (Maya Gupta, TEST_A)
- Median: 0.70
- Lowest: 0.42 (Mike Rodriguez — web dev, wrong domain)

**Ground truth comparison:**
- Expected top 3: Maya, Sarah, David
- Actual top 3: Maya, TEST_A, Sarah
- Missing from top 3: David (ranked #4 with 0.81)

---

**Next:** [V2: Deterministic Baseline](../v2-deterministic) →
