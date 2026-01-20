# Resume Matching Engine - Architecture Specification

> **Version**: 1.0  
> **Status**: DRAFT - Pending Review  
> **Last Updated**: 2026-01-19

---

## 1. Executive Summary

This document specifies the technical architecture for an AI-powered resume matching engine that scores resumes against job descriptions. The system uses a **hybrid multi-model approach** combining semantic understanding with explicit skill extraction to produce accurate, explainable relevance scores.

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
│                      PREPROCESSING LAYER                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ • Text normalization (lowercase, unicode handling)           │   │
│  │ • Section extraction (experience, skills, education)         │   │
│  │ • Noise removal (headers, footers, formatting artifacts)     │   │
│  │ • Skill entity extraction (NER or regex-based)               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SCORING LAYER                                 │
│                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐                   │
│  │   LLM Scorer        │  │   Embedding Scorer  │                   │
│  │   (Primary)         │  │   (Baseline)        │                   │
│  │                     │  │                     │                   │
│  │ • Structured Prompt │  │ • Bi-Encoder        │                   │
│  │ • Reasoning output  │  │ • Cosine similarity │                   │
│  │ • Skills extraction │  │ • High speed        │                   │
│  │                     │  │                     │                   │
│  │   Weight: 70%       │  │   Weight: 30%       │                   │
│  └──────────┬──────────┘  └──────────┬──────────┘                   │
│             │                        │                              │
│             └────────────────────────┼────────────────────┘         │
│                                      ▼                              │
│                        ┌─────────────────────────┐                  │
│                        │   Weighted Aggregation  │                  │
│                        │   Final Score: 0.0-1.0  │                  │
│                        └─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ {                                                            │   │
│  │   "resume_id": "resume_001",                                 │   │
│  │   "final_score": 0.82,                                       │   │
│  │   "breakdown": {                                             │   │
│  │     "semantic_score": 0.78,                                  │   │
│  │     "cross_encoder_score": 0.85,                             │   │
│  │     "skill_match_score": 0.80                                │   │
│  │   },                                                         │   │
│  │   "matched_skills": ["Python", "FastAPI", "ML"],             │   │
│  │   "missing_skills": ["Kubernetes"]                           │   │
│  │ }                                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Explainability** | Score breakdown + matched/missing skills |
| **Modularity** | Each scorer is independent, weights configurable |
| **Reproducibility** | Fixed random seeds, versioned models |
| **Simplicity** | No over-engineering, focused on POC quality |

---

## 3. Model Selection & Justification

### 3.1 Primary Model: LLM-Based Scorer (via Groq/Gemini)

| Attribute | Value |
|-----------|-------|
| **Model** | `llama-3.1-70b-versatile` (Groq) or `gemini-1.5-flash` |
| **Purpose** | Complex reasoning, skill extraction, and nuanced scoring |
| **Why?** | Directly demonstrates **Prompt Engineering** skills. Provides explainable results. |

**Key Advantages:**
- Understands context and seniority (e.g., "lead" vs "junior")
- Handles fuzzy skill matching natively
- Provides "Reasoning" which enhances the product UI/UX

### 3.2 Baseline Model: Bi-Encoder (Semantic Embeddings)

| Attribute | Value |
|-----------|-------|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Purpose** | Fast semantic similarity for comparison |
| **Weight** | 30% |

**Why?**
- Serves as a technical baseline to evaluate LLM performance
- Demonstrates breadth of AI engineering knowledge (traditional NLP vs LLMs)

---

## 4. Scoring Algorithm

### 4.1 Component Scores

```python
# Pseudo-code for scoring pipeline

def compute_final_score(job_description: str, resume: str) -> ScoringResult:
    # 1. LLM Scoring (Prompt Engineering Focus)
    llm_result = llm_scorer.score(job_description, resume)
    
    # 2. Embedding Similarity (Baseline)
    embedding_score = embedding_scorer.score(job_description, resume)
    
    # 3. Weighted Aggregation
    final_score = (0.70 * llm_result.score + 0.30 * embedding_score)
    
    return ScoringResult(
        final_score=final_score,
        llm_score=llm_result.score,
        embedding_score=embedding_score,
        reasoning=llm_result.reasoning,
        matched_skills=llm_result.matched_skills,
        missing_skills=llm_result.missing_skills
    )
```

### 4.2 Weight Rationale

| Component | Weight | Rationale |
|-----------|--------|-----------|
| LLM Scorer | 70% | Deep reasoning and semantic matching |
| Embedding Scorer | 30% | Fast semantic baseline for comparison |

> **Note**: Weights are configurable and should be tuned based on evaluation results.

---

## 5. Data Preprocessing Pipeline

### 5.1 Text Normalization

```python
def preprocess_text(text: str) -> str:
    """
    Standard text preprocessing for resumes and job descriptions.
    """
    # 1. Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Lowercase (preserving acronyms where needed)
    text = text.lower()
    
    # 3. Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Remove common noise patterns
    text = remove_email_phone(text)
    text = remove_urls(text)
    
    return text
```

### 5.2 Skill Extraction

```python
SKILL_TAXONOMY = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "go", "rust", "c++",
    
    # Frameworks
    "fastapi", "django", "flask", "react", "vue", "angular", "nextjs",
    
    # ML/AI
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
    "machine learning", "deep learning", "nlp", "computer vision",
    
    # Cloud/DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
    
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    
    # ... (full taxonomy in implementation)
}

def extract_skills(text: str) -> set[str]:
    """
    Extract skills from text using fuzzy matching against taxonomy.
    """
    text_lower = text.lower()
    found_skills = set()
    
    for skill in SKILL_TAXONOMY:
        if skill in text_lower:
            found_skills.add(skill)
        # Also check common variations (e.g., "ML" -> "machine learning")
    
    return found_skills
```

---

## 6. API Design

### 6.1 Core Interface

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ScoringResult:
    resume_id: str
    final_score: float      # Weighted average (0.0 to 1.0)
    llm_score: float        # Primary intelligence score
    embedding_score: float  # Baseline semantic similarity
    reasoning: str          # Qualitative justification
    matched_skills: List[str]
    missing_skills: List[str]

class ResumeMatchingEngine:
    """
    Main interface for the resume matching system.
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize models and configuration."""
        pass
    
    def score_resume(
        self, 
        job_description: str, 
        resume: str,
        resume_id: str = "resume"
    ) -> ScoringResult:
        """Score a single resume against a job description."""
        pass
    
    def score_resumes(
        self,
        job_description: str,
        resumes: List[tuple[str, str]]  # [(resume_id, resume_text), ...]
    ) -> List[ScoringResult]:
        """Score multiple resumes and return ranked results."""
        pass
    
    def rank_resumes(
        self,
        job_description: str,
        resumes: List[tuple[str, str]]
    ) -> List[ScoringResult]:
        """Score and rank resumes by final_score descending."""
        pass
```

### 6.2 Configuration

```python
@dataclass
class EngineConfig:
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "groq"  # or "gemini"
    llm_model: str = "llama-3.1-70b-versatile"
    
    # Scoring weights
    llm_weight: float = 0.70
    embedding_weight: float = 0.30
    
    # Processing settings
    max_resume_length: int = 8000  # characters
    temperature: float = 0.0      # For deterministic scoring
```

---

## 7. Project Structure

```
ema/
├── README.md                      # Main documentation
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup (optional)
│
├── docs/                          # Specifications
│   ├── 01-architecture-spec.md    # This document
│   ├── 02-data-spec.md            # Synthetic data specification
│   └── 03-evaluation-spec.md      # Evaluation methodology
│
├── src/
│   ├── __init__.py
│   ├── engine.py                  # ResumeMatchingEngine class
│   ├── scorers/
│   │   ├── __init__.py
│   │   ├── llm.py                 # LLM Scorer (Groq/Gemini)
│   │   └── semantic.py            # Bi-encoder baseline
│   ├── prompts.py                 # Centralized prompt templates
│   ├── preprocessing.py           # Text preprocessing
│   └── config.py                  # Configuration (API keys, weights)
│
├── data/
│   ├── job_descriptions/
│   │   └── ema_ai_apps_engineer.txt
│   ├── resumes/
│   │   ├── res_001_yashpreet.txt
│   │   ├── res_002_sarah.txt
│   │   ├── res_003_maya.txt
│   │   ├── res_004_david.txt
│   │   ├── res_005_mike.txt
│   │   ├── res_006_priya.txt
│   │   ├── res_007_sam.txt
│   │   ├── res_008_jordan.txt
│   │   ├── res_009_james.txt
│   │   └── res_010_elena.txt
│   └── labeled_dataset.json       # Ground truth labels
│
├── notebooks/
│   └── evaluation.ipynb           # Analysis and visualization
│
├── tests/
│   ├── test_engine.py
│   ├── test_preprocessing.py
│   └── test_skill_extraction.py
│
└── demo.py                        # Quick demo script
```

---

## 8. Dependencies

```
# Core
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0

# Text Processing
rapidfuzz>=3.0.0

# Data & Analysis
pandas>=2.0.0
numpy>=1.24.0

# Visualization (for notebook)
matplotlib>=3.7.0
seaborn>=0.12.0

# Development
pytest>=7.0.0
black>=23.0.0
```

---

## 9. Open Questions / Decisions Needed

1. **GPU Support**: Should we require/support GPU acceleration, or keep it CPU-only for simplicity?

2. **Skill Taxonomy Scope**: How comprehensive should the skill list be? Start with ~100 core tech skills or aim for ~500?

3. **Resume Sectioning**: Should we attempt to parse resume sections (experience, education, skills) separately, or treat the entire resume as one block of text?

4. **Score Normalization**: Should cross-encoder scores be calibrated/normalized if they don't naturally fall in [0, 1]?

---

## 10. Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Ranking Accuracy | Good matches ranked above Poor | Basic sanity check |
| Score Separation | Good (>0.7), Partial (0.4-0.7), Poor (<0.4) | Clear differentiation |
| Latency | <500ms per resume | Reasonable for POC |
| Code Coverage | >80% | Quality assurance |

---

*Document awaiting review before implementation.*
