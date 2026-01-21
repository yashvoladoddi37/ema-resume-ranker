# 🎯 AI-Powered Resume Matcher

A production-grade resume ranking engine using a **Two-Stage LLM Pipeline** for intelligent, explainable candidate matching.

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/yashvoladoddi37/ema-resume-ranker.git
cd ema-resume-ranker
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Run evaluation
python evaluate.py

# 4. Launch dashboard
streamlit run app.py
```

---

## 🏗️ Architecture: Two-Stage LLM Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TWO-STAGE LLM PIPELINE                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Resume.txt ──▶ [STAGE 1: PARSER] ──▶ Structured JSON                   │
│                      │                      │                            │
│                      │                      ▼                            │
│                      │               ┌──────────────┐                   │
│                      │               │ candidate    │                   │
│                      │               │ skills       │                   │
│                      │               │ experience[] │                   │
│                      │               │ domains      │                   │
│                      │               └──────────────┘                   │
│                      │                      │                            │
│                      ▼                      ▼                            │
│              [STAGE 2: SCORER] ◀── Job Description                      │
│                      │                                                   │
│                      ▼                                                   │
│     ┌────────────────┬────────────────┬────────────────┐                │
│     │ skill_match    │ experience_    │ domain_fit    │                │
│     │ (50%)          │ depth (30%)    │ (20%)         │                │
│     └────────────────┴────────────────┴────────────────┘                │
│                      │                                                   │
│                      ▼                                                   │
│           Final Score = Σ(dimension × weight)                            │
│                      │                                                   │
│                      ▼                                                   │
│           [RANK BY SCORE] ──▶ Sorted Results                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Why Two Stages?

| Stage | Purpose | Output |
|-------|---------|--------|
| **Parser** | Extract structured data from messy resume text | JSON with skills, experience, education |
| **Scorer** | Evaluate structured data against job requirements | Per-dimension scores with reasoning |

This separation ensures:
1. **Reliable extraction** — Parser focuses only on data extraction
2. **Fair scoring** — Scorer works on structured data, not raw text
3. **Full auditability** — Every step is logged and inspectable

---

## 📊 Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Skill Match** | 50% | Coverage of required + preferred skills |
| **Experience Depth** | 30% | Years + relevance of experience |
| **Domain Fit** | 20% | AI/ML + Support domain alignment |

### Example Output
```
#1 | Maya Gupta — Score: 0.850
    ├── Skill Match:      0.80 → "Python, LangChain, RAG found"
    ├── Experience Depth: 0.90 → "4 years, 2.5 in AI solutions"
    └── Domain Fit:       0.90 → "Direct GenAI + customer-facing"
```

---

## 📈 Evaluation Metrics

We treat resume ranking as an **Information Retrieval** problem, not classification.

| Metric | Score | Target | Description |
|--------|-------|--------|-------------|
| **nDCG@3** | 0.837 | ≥0.85 | Top 3 ranking quality |
| **Precision@1** | 1.000 | 1.00 | Is #1 actually a good match? |
| **Recall@3** | 0.667 | ≥0.90 | Are all good candidates in top 3? |

### Why These Metrics?

- **nDCG@3**: Measures if the best candidates are ranked highest
- **Precision@1**: Hiring managers look at the top candidate first
- **Recall@3**: We don't want to miss qualified candidates

---

## 🔍 Audit Trail

Every evaluation run saves full LLM I/O for debugging:

```bash
python evaluate_with_logging.py
```

Creates timestamped folders:
```
runs/run_20260121_131947/
├── 01_raw_resumes/          # Original resume text
├── 02_parser_prompts/       # LLM prompts for parsing
├── 03_parsed_data/          # Parser LLM outputs
├── 04_scorer_prompts/       # LLM prompts for scoring
├── 05_scorer_outputs/       # Scorer LLM outputs
├── 06_final_results/        # Per-candidate final results
├── all_results.json         # Combined ranked results
└── metrics.json             # Evaluation metrics
```

---

## 🤔 Why Not Embeddings?

We **deliberately chose LLM-based scoring** over vector embeddings because:

| Embeddings | LLM Scoring |
|------------|-------------|
| Measures **similarity** | Measures **suitability** |
| "Java Dev" ≈ "Python Dev" | "Java Dev" ≠ "Python Dev" for Python role |
| Can't count years | Can reason: "3+ years required" |
| Can't explain WHY | Returns structured reasoning |

**For retrieval** (find top 50 from 10,000), embeddings are great.
**For ranking/evaluation** (compare 10 candidates), LLM reasoning is superior.

---

## 📂 Project Structure

```
ema-resume-ranker/
├── src/
│   ├── resume_parser.py     # Stage 1: Structured extraction
│   ├── resume_scorer.py     # Stage 2: Dimension scoring
│   ├── matching_engine.py   # Pipeline orchestrator
│   └── utils.py             # Metrics & utilities
├── data/
│   ├── resumes/             # 12 sample resumes
│   ├── job_descriptions/    # Target job posting
│   └── ground_truth.json    # Manual labels
├── runs/                    # Audit trail logs
├── app.py                   # Streamlit dashboard
├── evaluate.py              # Basic evaluation
└── evaluate_with_logging.py # Evaluation with full audit trail
```

---

## ⚠️ Known Limitations

1. **Experience scoring weights years heavily** — Candidates with fewer but highly relevant years may score lower
2. **Skill matching is keyword-based** — Synonyms may need explicit handling
3. **Single job description** — Currently optimized for AI Applications Engineer role

### Future Improvements

- Add "experience relevance multiplier" for domain-specific work
- Semantic skill matching with embeddings as a pre-filter
- A/B test different weight configurations with larger labeled dataset
- Fine-tune smaller models for lower latency

---

## 🛠️ Configuration

### Environment Variables
```bash
GROQ_API_KEY=your_key_here
MODEL_NAME=llama-3.3-70b-versatile  # Optional
TEMPERATURE=0                        # Optional
```

### Adjusting Weights

```python
from src.matching_engine import TwoStageMatchingEngine

engine = TwoStageMatchingEngine()
engine.update_weights(
    skill=0.40,      # Reduce skill weight
    experience=0.40, # Increase experience weight
    domain=0.20
)
```

Or use the **Streamlit dashboard** sliders for interactive weight tuning.

---

## 📝 License

MIT License — Built for the Ema AI Team.
