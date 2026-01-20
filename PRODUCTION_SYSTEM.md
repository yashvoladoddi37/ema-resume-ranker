# Production-Grade Hybrid Resume Matcher

## 🎯 System Architecture

### **Hybrid Scoring Approach (60% LLM + 40% Deterministic)**

This is a **production-ready** system that combines:

1. **LLM Intelligence (60% weight)** - Contextual understanding, nuanced reasoning
2. **Deterministic Rules (40% weight)** - Verifiable, auditable metrics
3. **Human-in-the-Loop (HITL)** - Override capabilities for edge cases

---

## 🔧 Deterministic Components

### 1. **Experience Extractor**
**Purpose**: Calculate total years of experience using multiple heuristics

**Methods**:
- Pattern matching: "5 years", "5+ years", "5 yrs"
- Date range calculation: "2020 - 2025" = 5 years
- Present date handling: "2020 - Present" = 6 years (as of 2026)

**Output**: `float` (e.g., 4.5 years)

**Scoring**: Linear scale where 3+ years = 1.0

---

### 2. **Skill Extractor**
**Purpose**: Rule-based skill matching against Ema-specific taxonomy

**Skill Categories**:
- **Required Skills** (8 total): python, api, rest, json, troubleshooting, production, technical support, saas
- **Preferred Skills** (14 total): genai, llm, ml, langchain, prompt engineering, observability, logging, dashboard, aws, gcp, crm, ats, soap, integration

**Method**: Exact word boundary matching using regex

**Output**:
```python
SkillProfile(
    matched_required={'python', 'api', 'rest'},
    matched_preferred={'genai', 'llm'},
    missing_required={'saas', 'technical support'},
    skill_coverage_score=0.545  # 54.5%
)
```

**Scoring**: 70% required coverage + 30% preferred coverage

---

### 3. **Domain Relevance Calculator**
**Purpose**: Measure keyword density for AI and Support domains

**AI Keywords**: ai, machine learning, llm, genai, langchain, rag, embedding, etc.
**Support Keywords**: support, troubleshooting, debugging, production issues, customer-facing, etc.

**Method**: Count keyword occurrences / total words × 10 (capped at 100%)

**Output**:
```python
{
    'ai_relevance': 0.85,      # 85% AI keyword density
    'support_relevance': 0.12  # 12% support keyword density
}
```

---

### 4. **Deterministic Score Aggregation**
**Formula**:
```
Deterministic Score = 
    20% × Experience Score +
    40% × Skill Coverage +
    20% × AI Relevance +
    20% × Support Relevance
```

**Example Breakdown**:
```
Experience: 4 years → 1.0 (capped at 3 years)
Skills: 6/11 matched → 0.545
AI Relevance: High keyword density → 1.0
Support Relevance: Low keyword density → 0.0
───────────────────────────────────────
Deterministic Score = 0.20×1.0 + 0.40×0.545 + 0.20×1.0 + 0.20×0.0 = 0.618
```

---

## 🧠 LLM Component

**Model**: Groq llama-3.3-70b-versatile  
**Temperature**: 0.0 (deterministic)  
**Output**: Structured JSON with score, reasoning, matched/missing skills

**Strengths**:
- Contextual understanding (e.g., "User of AI" vs "Builder of AI")
- Nuanced seniority assessment
- Soft skill evaluation
- Domain expertise recognition

---

## 🔀 Hybrid Combination

**Final Score Formula**:
```
Final Score = 0.6 × LLM Score + 0.4 × Deterministic Score
```

**Example**:
```
Candidate: Maya Gupta
LLM Score: 0.90 (excellent GenAI experience, strong reasoning)
Deterministic: 0.62 (good skills, high AI relevance, missing support keywords)
───────────────────────────────────────
Final Score = 0.6×0.90 + 0.4×0.62 = 0.79
```

---

## 👤 Human-in-the-Loop (HITL)

### **Override Mechanism**
```python
scorer.apply_hitl_override(
    result=candidate_result,
    override_score=0.95,
    reason="Candidate has unique startup experience in AI agents not captured by rules"
)
```

**Use Cases**:
1. Unique experience not in taxonomy (e.g., specific AI frameworks)
2. LLM hallucination or context miss
3. Manual review reveals critical insight
4. Edge cases requiring human judgment

**Audit Trail**: All overrides logged with reason and original score

---

## 📊 Production Results

### Rankings (Hybrid 60/40)
| Rank | Candidate | Final | LLM | Deterministic |
|------|-----------|-------|-----|---------------|
| 1 | Maya Gupta | 0.79 | 0.90 | 0.62 |
| 2 | Sarah Johnson | 0.72 | 0.80 | 0.61 |
| 3 | David Kim | 0.66 | 0.80 | 0.44 |
| 4 | Yashpreet | 0.60 | 0.70 | 0.44 |

### Key Insights
- **Maya**: High LLM (0.90) due to GenAI experience, moderate deterministic (0.62) due to missing support keywords
- **Sarah**: Balanced scores - strong support background recognized by both components
- **Yashpreet**: LLM recognizes AI skills (0.70), deterministic penalizes limited experience years (0.44)

---

## ✅ Production Advantages

1. **Auditability**: Every score has verifiable deterministic component
2. **Transparency**: Full breakdown of LLM + rules visible
3. **Reliability**: Deterministic floor prevents LLM hallucination
4. **Flexibility**: Adjustable weights (60/40 configurable)
5. **HITL Ready**: Override mechanism for edge cases
6. **Reproducibility**: Deterministic component always consistent

---

## 🚀 Usage

```bash
# Run production hybrid scorer
python demo_production.py

# Output: results_hybrid.json with full breakdowns
```

**Result Structure**:
```json
{
  "id": "res_003_maya",
  "final_score": 0.787,
  "llm_component": {
    "score": 0.9,
    "reasoning": "...",
    "matched_skills": [...],
    "missing_skills": [...]
  },
  "deterministic_component": {
    "score": 0.618,
    "years_experience": 4,
    "skill_coverage": 0.545,
    "ai_relevance": 1.0,
    "support_relevance": 0.0,
    "matched_required_skills": [...],
    "missing_required_skills": [...]
  },
  "hitl_override": null,
  "audit_trail": {...}
}
```

---

**This is enterprise-grade, production-ready AI evaluation.**
