# Resume Matching Engine - Architecture Specification

> **Version**: 2.0  
> **Status**: IMPLEMENTED  
> **Last Updated**: 2026-01-20

---

## 1. Executive Summary

This document specifies the technical architecture for an AI-powered resume matching engine that scores resumes against job descriptions. The system uses a **Sequential Hybrid Scoring** architecture combining LLM-based contextual reasoning (60%) with deterministic rule-based extraction (40%) to produce accurate, explainable, and auditable relevance scores.

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                │
│  ┌─────────────────┐              ┌─────────────────────────────┐   │
│  │ Job Description │              │ Resume(s)                   │   │
│  │ (Text/File)     │              │ (Text/Files - batch support)│   │
│  └────────┬────────┘              └─────────────┬───────────────┘   │
└───────────┼─────────────────────────────────────┼───────────────────┘
            │                                     │
            ▼                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SEQUENTIAL HYBRID ENGINE                          │
│                                                                     │
│  Step 1: Deterministic Scorer (40% weight)                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Regex-based skill extraction (Required + Preferred)        │  │
│  │ • Years of experience calculation (date parsing)             │  │
│  │ • AI keyword density (GenAI, LLM, LangChain, etc.)           │  │
│  │ • Support keyword density (troubleshooting, debugging)       │  │
│  │ • Education detection (CS, Engineering degrees)              │  │
│  │                                                              │  │
│  │ Output: Verifiable baseline score + extracted facts          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  Step 2: LLM Scorer (60% weight)                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Model: Llama 3.3-70B (via Groq API)                          │  │
│  │ • Receives deterministic facts as "ground truth" context     │  │
│  │ • Performs nuanced evaluation (seniority, domain fit)        │  │
│  │ • Generates technical breakdown:                             │  │
│  │   - Skill Alignment (0.0-1.0)                                │  │
│  │   - Experience Depth (0.0-1.0)                               │  │
│  │   - Domain Fit (0.0-1.0)                                     │  │
│  │ • Provides natural language reasoning                        │  │
│  │                                                              │  │
│  │ Output: Contextual score + reasoning + matched/missing skills│  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  Step 3: Weighted Aggregation                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Final Score = (0.6 × LLM) + (0.4 × Deterministic)            │  │
│  │ + Score explanation in natural language                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ {                                                            │   │
│  │   "id": "resume_001",                                        │   │
│  │   "final_score": 0.541,                                      │   │
│  │   "score_formula": "(0.6 × 0.62) + (0.4 × 0.423) = 0.541",   │   │
│  │   "score_explanation": "Natural language breakdown...",      │   │
│  │   "components": {                                            │   │
│  │     "llm": {                                                 │   │
│  │       "score": 0.62,                                         │   │
│  │       "reasoning": "Strong AI/ML background...",             │   │
│  │       "technical_breakdown": {                               │   │
│  │         "skill_alignment": 0.70,                             │   │
│  │         "experience_depth": 0.50,                            │   │
│  │         "domain_fit": 0.60                                   │   │
│  │       },                                                     │   │
│  │       "matched_skills": ["python", "llm", "langchain"],      │   │
│  │       "missing_skills": ["troubleshooting", "saas"]          │   │
│  │     },                                                       │   │
│  │     "deterministic": {                                       │   │
│  │       "score": 0.423,                                        │   │
│  │       "years_experience": 1.0,                               │   │
│  │       "required_coverage_pct": 37.5,                         │   │
│  │       "preferred_coverage_pct": 42.9                         │   │
│  │     }                                                        │   │
│  │   }                                                          │   │
│  │ }                                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Explainability** | Natural language score explanation + component breakdown |
| **Auditability** | Deterministic baseline provides verifiable facts |
| **Transparency** | Every score shows exact formula and weights |
| **Reproducibility** | Temperature=0 for LLM, deterministic regex patterns |

---

## 3. Model Selection & Justification

### 3.1 Primary Model: Llama 3.3-70B (via Groq)

| Attribute | Value |
|-----------|-------|
| **Model** | `llama-3.3-70b-versatile` |
| **Provider** | Groq (LPU inference) |
| **Purpose** | Contextual reasoning, skill evaluation, nuanced scoring |
| **Weight** | 60% |

**Why Llama 3.3-70B?**
- Strong instruction-following for structured JSON output
- Excellent at nuanced reasoning (differentiates "senior generalist" vs "domain expert")
- Fast inference via Groq (~2s per resume vs ~10s on OpenAI)
- Cost-effective: $0.59/M tokens vs GPT-4's $30/M (50x cheaper)

**Why Groq?**
- LPU (Language Processing Unit) provides 500 tokens/sec throughput
- Free tier: 14,400 tokens/min for testing
- Native JSON mode support

### 3.2 Deterministic Scorer (Rule-Based)

| Attribute | Value |
|-----------|-------|
| **Method** | Regex patterns + keyword matching |
| **Purpose** | Verifiable baseline, fact extraction |
| **Weight** | 40% |

**Why Deterministic Component?**
- Provides auditable "ground truth" (exact skill matches, years extracted)
- No API costs, instant execution
- Anchors LLM evaluation in verifiable facts
- Enables debugging (can trace why a score was given)

### 3.3 Why NOT Embeddings (SBERT)?

**Embeddings were deliberately avoided** for the core scoring engine:

1. **Similarity ≠ Suitability**: Embeddings measure semantic proximity. "Senior Java Developer" is highly similar to "Senior Python Developer" in vector space, but for hiring, this is a disqualification.

2. **No Reasoning**: Embeddings can't perform calculations ("Is 2 years enough for 3+ years requirement?") or apply business logic ("Is LangChain more valuable than Spring Boot for this role?").

3. **Use Case Fit**: Embeddings are ideal for **Retrieval** (finding top 50 from 10,000). For **Evaluation** (ranking and scoring a shortlist), LLM + Deterministic provides superior precision and explainability.

---

## 4. Scoring Algorithm

### 4.1 Deterministic Component (40%)

**Weighting Breakdown:**
- Technical Skills: 35% (API, Python, REST, JSON matching)
- AI Relevance: 35% (GenAI/LLM keyword density)
- Experience: 15% (Years calculated, target: 3+)
- Education: 10% (Relevant CS/Engineering degree)
- Support Relevance: 5% (Troubleshooting keyword density)

**Total: 100%**

**Skill Taxonomy:**
```python
REQUIRED_SKILLS = {
    'python', 'api', 'rest', 'json', 'troubleshooting',
    'production', 'technical support', 'saas'
}

PREFERRED_SKILLS = {
    'genai', 'llm', 'ml', 'langchain', 'prompt engineering',
    'observability', 'logging', 'dashboard', 'aws', 'gcp',
    'crm', 'ats', 'soap', 'integration'
}

AI_KEYWORDS = {
    'ai', 'artificial intelligence', 'machine learning', 'ml',
    'llm', 'large language model', 'genai', 'generative ai',
    'langchain', 'langgraph', 'prompt', 'rag', 'embedding'
}

SUPPORT_KEYWORDS = {
    'support', 'customer success', 'technical support',
    'troubleshooting', 'debugging', 'production issues',
    'incident', 'ticket', 'escalation', 'customer-facing'
}
```

### 4.2 LLM Component (60%)

**Prompt Strategy:**
- Sequential enrichment: LLM receives deterministic facts as context
- Temperature=0 for deterministic behavior
- JSON mode for structured output
- Explicit instruction to ignore institution prestige

**Output Structure:**
```json
{
  "score": 0.62,
  "reasoning": "Qualitative assessment...",
  "technical_breakdown": {
    "skill_alignment": 0.70,
    "experience_depth": 0.50,
    "domain_fit": 0.60
  },
  "matched_skills": ["python", "llm"],
  "missing_skills": ["troubleshooting"]
}
```

### 4.3 Final Aggregation

```python
final_score = (0.6 * llm_score) + (0.4 * deterministic_score)

# Natural language explanation generated:
explanation = f"""
**Final Score: {final_score:.4f}**

1. LLM Evaluation (60%): {llm_score:.2f}
   - Based on skill alignment, experience depth, domain fit
   
2. Deterministic Evaluation (40%): {det_score:.3f}
   - Technical Skills (35%): {matched_required}/8 required skills
   - AI Relevance (35%): Keyword analysis
   - Experience (15%): {years} years (Target: 3+)
   - Education (10%): {degree_status}
   - Support (5%): Keyword density
"""
```

---

## 5. Evaluation Metrics

### 5.1 Information Retrieval Metrics

| Metric | Score | Target | Interpretation |
|:-------|:------|:-------|:---------------|
| **nDCG@3** | 0.954 | ≥0.85 | Ranking quality of top 3 |
| **Precision@1** | 100% | 100% | Top candidate is always good |
| **Recall@3** | 100% | 100% | All good candidates in top 3 |
| **Pairwise Accuracy** | 94.7% | ≥85% | Correct ordering frequency |

**Why These Metrics?**
- Resume ranking is an **IR problem**, not classification
- nDCG measures ranking quality with position-based weighting
- Precision@1 ensures top candidate is always qualified
- Recall@3 ensures no good candidates are buried

---

## 6. Project Structure

```
ema/
├── README.md                      # Main documentation
├── requirements.txt               # Python dependencies
├── assignment-statement.txt       # Original assignment
│
├── docs/
│   └── 01-architecture-spec.md    # This document
│
├── app.py                         # Streamlit interactive dashboard
├── demo_hybrid.py                 # Core engine + evaluation logic
│
├── data/
│   ├── job_descriptions/
│   │   └── ema_ai_apps_engineer.txt
│   ├── resumes/
│   │   ├── res_001_yashpreet.txt
│   │   ├── res_002_sarah.txt
│   │   └── ... (12 total)
│   └── labeled_dataset.json       # Ground truth labels
│
├── results_hybrid.json            # Cached evaluation results
└── .env                           # API keys (gitignored)
```

---

## 7. Dependencies

```
# Core
groq>=0.4.0                    # LLM API
python-dotenv>=1.0.0           # Environment management

# UI
streamlit>=1.30.0              # Interactive dashboard
plotly>=5.18.0                 # Visualizations

# Evaluation
scikit-learn>=1.3.0            # nDCG, metrics

# Development
pytest>=7.0.0                  # Testing
```

---

## 8. API Design

### 8.1 Core Classes

```python
class DeterministicScorer:
    """Rule-based extraction and scoring."""
    
    def score(self, resume_text: str) -> Dict[str, Any]:
        """
        Returns:
        {
            'score': float,
            'breakdown': {...},
            'extracted_data': {
                'years_experience': float,
                'matched_required': List[str],
                'matched_preferred': List[str],
                'missing_required': List[str]
            }
        }
        """

class LLMScorer:
    """LLM-based contextual evaluation."""
    
    def score(
        self, 
        job_description: str, 
        resume_text: str,
        deterministic_context: Dict = None
    ) -> Dict[str, Any]:
        """
        Returns:
        {
            'score': float,
            'reasoning': str,
            'technical_breakdown': {
                'skill_alignment': float,
                'experience_depth': float,
                'domain_fit': float
            },
            'matched_skills': List[str],
            'missing_skills': List[str]
        }
        """

class HybridEngine:
    """Production-grade resume matching."""
    
    def evaluate(
        self, 
        job_description: str, 
        resume: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Returns:
        {
            'id': str,
            'final_score': float,
            'score_formula': str,
            'score_explanation': str,  # Natural language
            'components': {
                'llm': {...},
                'deterministic': {...}
            }
        }
        """
    
    def rank_all(
        self,
        job_description: str,
        resumes: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Evaluate and rank all resumes."""
```

---

## 9. Key Features

### 9.1 Interactive Streamlit Dashboard

- Resume upload functionality (.txt files)
- Live evaluation with progress indicators
- Job description display
- Metrics visualization (nDCG, Precision, Recall)
- Individual candidate analysis with tabs:
  - AI Reasoning
  - Score Logic (natural language explanation)
  - Raw Components (JSON)
  - Resume Text

### 9.2 Production Features

- **API Key Fallback**: Automatic switch to `GROQ_API_KEY_2` on rate limits
- **Rate Limiting**: `time.sleep(1)` between API calls
- **Error Handling**: Graceful degradation on API failures
- **Caching**: Results saved to `results_hybrid.json`

---

## 10. Success Criteria

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| nDCG@3 | ≥0.85 | 0.954 | ✅ |
| Precision@1 | 100% | 100% | ✅ |
| Recall@3 | 100% | 100% | ✅ |
| Pairwise Accuracy | ≥85% | 94.7% | ✅ |
| Latency | <5s per resume | ~2s | ✅ |
| Explainability | Required | Full breakdown | ✅ |

---

## 11. Future Improvements

1. **LLM-based Experience Parsing**: Replace regex with LLM to handle "April 2024 - Present" correctly
2. **Remove Education Weight**: JD doesn't mention degree requirements (currently 10%)
3. **Embedding Pre-filter**: For 1000+ resumes, add SBERT retrieval before LLM evaluation
4. **Fine-tuning**: Train smaller model on labeled data to reduce API costs
5. **A/B Testing**: Experiment with different LLM prompts and weights

---

*Document reflects production implementation as of 2026-01-20.*
