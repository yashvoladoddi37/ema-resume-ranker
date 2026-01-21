# Technical Specification: Two-Stage LLM Resume Matching Engine

**Target:** Complete, working system in 24 hours  
**Stack:** Python 3.10+, Groq API, Streamlit  
**Scale:** 10-20 resumes (POC)

---

## 1. Project Structure

```
ema-resume-matcher/
├── README.md                          # Main documentation
├── requirements.txt                   # Dependencies
├── .env.example                       # API key template
├── .gitignore                        # Git ignore file
│
├── src/
│   ├── __init__.py
│   ├── resume_parser.py              # Stage 1: Parsing
│   ├── resume_scorer.py              # Stage 2: Scoring
│   ├── matching_engine.py            # Stage 3: Aggregation
│   └── utils.py                      # Helper functions
│
├── data/
│   ├── job_descriptions/
│   │   └── ema_ai_apps_engineer.txt  # The JD
│   ├── resumes/
│   │   ├── resume_001.txt            # 10-20 sample resumes
│   │   ├── resume_002.txt
│   │   └── ...
│   └── ground_truth.json             # Manual labels
│
├── notebooks/
│   └── evaluation.ipynb              # Evaluation experiments
│
├── tests/
│   ├── test_parser.py
│   ├── test_scorer.py
│   └── test_engine.py
│
├── app.py                            # Streamlit dashboard
├── evaluate.py                       # Run full evaluation
└── results.json                      # Cached results
```

---

## 2. Dependencies (`requirements.txt`)

```txt
# Core
groq>=0.4.0
python-dotenv>=1.0.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0

# Evaluation
scikit-learn>=1.3.0

# UI
streamlit>=1.30.0
plotly>=5.18.0

# Development
pytest>=7.4.0
black>=23.0.0
```

---

## 3. Environment Setup (`.env.example`)

```bash
# Groq API Keys
GROQ_API_KEY=your_primary_key_here
GROQ_API_KEY_2=your_backup_key_here

# Model Configuration
MODEL_NAME=llama-3.3-70b-versatile
TEMPERATURE=0
MAX_TOKENS=2048
```

---

## 4. Core Component Specifications

### **4.1 Stage 1: Resume Parser (`src/resume_parser.py`)**

```python
"""
Resume Parser - Stage 1 of Two-Stage Pipeline
Extracts structured data from raw resume text using LLM.
"""

from groq import Groq
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv

class ResumeParser:
    """
    Parses raw resume text into structured JSON format.
    
    Handles:
    - Multiple date formats ("2020-2022", "Jan 2020 - Present", etc.)
    - Skill extraction with synonyms
    - Experience calculation
    - Education parsing
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize parser with Groq API.
        
        Args:
            api_key: Groq API key. If None, loads from environment.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        self.temperature = float(os.getenv("TEMPERATURE", 0))
    
    def parse(self, resume_text: str) -> Dict[str, Any]:
        """
        Extract structured data from resume.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            {
                "candidate_name": str,
                "skills": List[str],
                "total_years_experience": float,
                "experience": [
                    {
                        "title": str,
                        "company": str,
                        "duration_years": float,
                        "start_date": str,
                        "end_date": str,
                        "key_responsibilities": List[str]
                    }
                ],
                "education": {
                    "degree": str,
                    "field": str,
                    "institution": str
                },
                "certifications": List[str],
                "domains": List[str]  # e.g., ["AI/ML", "Customer Support"]
            }
        """
        
        prompt = self._build_parsing_prompt(resume_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                max_tokens=2048
            )
            
            parsed_data = json.loads(response.choices[0].message.content)
            return self._validate_parsed_data(parsed_data)
            
        except Exception as e:
            print(f"❌ Parsing failed: {e}")
            return self._get_empty_structure()
    
    def _build_parsing_prompt(self, resume_text: str) -> str:
        """Build the LLM prompt for parsing."""
        
        return f"""You are a resume parser. Extract structured information from the following resume.

RESUME TEXT:
{resume_text}

TASK:
Extract and return ONLY a valid JSON object with this EXACT structure:

{{
  "candidate_name": "Full Name",
  "skills": ["skill1", "skill2", "skill3"],
  "total_years_experience": 0.0,
  "experience": [
    {{
      "title": "Job Title",
      "company": "Company Name",
      "duration_years": 0.0,
      "start_date": "Month Year",
      "end_date": "Month Year or Present",
      "key_responsibilities": ["responsibility1", "responsibility2"]
    }}
  ],
  "education": {{
    "degree": "Degree Name",
    "field": "Field of Study",
    "institution": "University Name"
  }},
  "certifications": ["cert1", "cert2"],
  "domains": ["domain1", "domain2"]
}}

CRITICAL INSTRUCTIONS:
1. **Date Handling:**
   - If end_date is "Present", "Current", or missing, use "January 2025"
   - Calculate duration_years accurately (e.g., "Jan 2023 - Jan 2025" = 2.0 years)
   - For partial years, use decimals (e.g., "Apr 2024 - Jan 2025" = 0.75 years)

2. **Skills Extraction:**
   - Extract ALL technical skills (languages, frameworks, tools, platforms)
   - Include synonyms (e.g., if resume says "GenAI", include "Generative AI", "LLM")
   - Include both explicitly stated and implied skills

3. **Total Experience:**
   - Sum all duration_years from experience array
   - Do NOT count overlapping periods twice

4. **Domains:**
   - Infer from experience (e.g., "built chatbot" → "Conversational AI")
   - Common domains: "AI/ML", "Customer Support", "SaaS", "Cloud Infrastructure"

5. **Output Format:**
   - Return ONLY valid JSON
   - NO markdown, NO backticks, NO explanations
   - All string fields must use double quotes
   - Use null for missing optional fields

EXAMPLES OF CORRECT DATE PARSING:
- "2020 - 2022" → duration_years: 2.0
- "Jan 2023 - Present" → duration_years: 2.0 (as of Jan 2025)
- "April 2024 - Current" → duration_years: 0.75

Begin parsing now:"""
    
    def _validate_parsed_data(self, data: Dict) -> Dict:
        """Validate and clean parsed data."""
        
        required_fields = [
            "candidate_name", "skills", "total_years_experience",
            "experience", "education", "certifications", "domains"
        ]
        
        for field in required_fields:
            if field not in data:
                print(f"⚠️  Missing field: {field}, using default")
                data[field] = self._get_default_value(field)
        
        # Ensure total_years is a float
        data["total_years_experience"] = float(data["total_years_experience"])
        
        return data
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing field."""
        defaults = {
            "candidate_name": "Unknown",
            "skills": [],
            "total_years_experience": 0.0,
            "experience": [],
            "education": {"degree": None, "field": None, "institution": None},
            "certifications": [],
            "domains": []
        }
        return defaults.get(field, None)
    
    def _get_empty_structure(self) -> Dict:
        """Return empty structure on parsing failure."""
        return {
            "candidate_name": "Parse Failed",
            "skills": [],
            "total_years_experience": 0.0,
            "experience": [],
            "education": {"degree": None, "field": None, "institution": None},
            "certifications": [],
            "domains": []
        }
```

---

### **4.2 Stage 2: Resume Scorer (`src/resume_scorer.py`)**

```python
"""
Resume Scorer - Stage 2 of Two-Stage Pipeline
Scores parsed resume against job description on multiple dimensions.
"""

from groq import Groq
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv

class ResumeScorer:
    """
    Scores parsed resume data against job description.
    
    Dimensions:
    - skill_match: How well skills align (0.0-1.0)
    - experience_depth: Years + relevance (0.0-1.0)
    - domain_fit: Background alignment (0.0-1.0)
    """
    
    def __init__(self, api_key: str = None):
        """Initialize scorer with Groq API."""
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        self.temperature = float(os.getenv("TEMPERATURE", 0))
    
    def score(
        self, 
        job_description: str, 
        parsed_resume: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score resume on multiple dimensions.
        
        Args:
            job_description: The job posting text
            parsed_resume: Output from ResumeParser.parse()
            
        Returns:
            {
                "skill_match": {
                    "score": 0.0-1.0,
                    "reasoning": "explanation",
                    "matched_skills": ["skill1", "skill2"],
                    "missing_skills": ["skill3", "skill4"]
                },
                "experience_depth": {
                    "score": 0.0-1.0,
                    "reasoning": "explanation"
                },
                "domain_fit": {
                    "score": 0.0-1.0,
                    "reasoning": "explanation"
                },
                "overall_assessment": "summary"
            }
        """
        
        prompt = self._build_scoring_prompt(job_description, parsed_resume)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                max_tokens=2048
            )
            
            scores = json.loads(response.choices[0].message.content)
            return self._validate_scores(scores)
            
        except Exception as e:
            print(f"❌ Scoring failed: {e}")
            return self._get_default_scores()
    
    def _build_scoring_prompt(
        self, 
        jd: str, 
        parsed: Dict
    ) -> str:
        """Build the LLM prompt for scoring."""
        
        return f"""You are evaluating a candidate for a job opening.

JOB DESCRIPTION:
{jd}

CANDIDATE PROFILE (extracted from resume):
{json.dumps(parsed, indent=2)}

TASK:
Evaluate the candidate on three dimensions and return ONLY a JSON object:

{{
  "skill_match": {{
    "score": 0.0,
    "reasoning": "Brief explanation of skill alignment",
    "matched_skills": ["skill1", "skill2"],
    "missing_skills": ["skill3", "skill4"]
  }},
  "experience_depth": {{
    "score": 0.0,
    "reasoning": "Brief explanation including years comparison"
  }},
  "domain_fit": {{
    "score": 0.0,
    "reasoning": "Brief explanation of domain alignment"
  }},
  "overall_assessment": "1-2 sentence summary of candidacy"
}}

SCORING GUIDELINES:

1. **skill_match (0.0-1.0):**
   - 1.0 = Has ALL required skills + most preferred skills
   - 0.8 = Has ALL required skills, some preferred
   - 0.6 = Has MOST required skills
   - 0.4 = Has SOME required skills
   - 0.2 = Has FEW required skills
   - 0.0 = Has NO required skills
   
   - matched_skills: List skills from candidate that match JD requirements
   - missing_skills: List important skills from JD the candidate lacks

2. **experience_depth (0.0-1.0):**
   - Consider BOTH years AND relevance
   - If JD requires "3+ years" and candidate has 3+: score ≥ 0.7
   - If JD requires "3+ years" and candidate has 2 years: score ≤ 0.6
   - Relevant experience is worth more than total years
   - Senior roles with junior experience: score ≤ 0.5

3. **domain_fit (0.0-1.0):**
   - How well does candidate's background align with role domain?
   - For AI Applications Engineer: AI/ML background + customer-facing = high
   - Related domains (e.g., ML Engineer → AI Engineer) = medium-high
   - Unrelated domains = low

CRITICAL RULES:
- Use the FULL 0.0-1.0 scale (not just 0.6-0.9)
- Be critical but fair
- NO markdown formatting
- Return ONLY valid JSON
- Each score must be a float with 2 decimal places
- Reasoning should be 1-2 sentences max per dimension

Begin evaluation now:"""
    
    def _validate_scores(self, scores: Dict) -> Dict:
        """Validate score structure and ranges."""
        
        required_dims = ["skill_match", "experience_depth", "domain_fit"]
        
        for dim in required_dims:
            if dim not in scores:
                print(f"⚠️  Missing dimension: {dim}")
                scores[dim] = {"score": 0.5, "reasoning": "Not evaluated"}
            
            # Ensure score is float between 0 and 1
            if "score" in scores[dim]:
                score = float(scores[dim]["score"])
                scores[dim]["score"] = max(0.0, min(1.0, score))
        
        # Ensure skill_match has lists
        if "matched_skills" not in scores["skill_match"]:
            scores["skill_match"]["matched_skills"] = []
        if "missing_skills" not in scores["skill_match"]:
            scores["skill_match"]["missing_skills"] = []
        
        return scores
    
    def _get_default_scores(self) -> Dict:
        """Return default scores on failure."""
        return {
            "skill_match": {
                "score": 0.5,
                "reasoning": "Scoring failed",
                "matched_skills": [],
                "missing_skills": []
            },
            "experience_depth": {
                "score": 0.5,
                "reasoning": "Scoring failed"
            },
            "domain_fit": {
                "score": 0.5,
                "reasoning": "Scoring failed"
            },
            "overall_assessment": "Evaluation incomplete due to system error"
        }
```

---

### **4.3 Stage 3: Matching Engine (`src/matching_engine.py`)**

```python
"""
Matching Engine - Orchestrates two-stage pipeline and aggregates scores.
"""

import time
from typing import Dict, List, Any
from .resume_parser import ResumeParser
from .resume_scorer import ResumeScorer

class TwoStageMatchingEngine:
    """
    Complete resume matching system.
    
    Pipeline:
    1. Parse resume → structured data
    2. Score against JD → dimension scores
    3. Aggregate → final score with explanation
    """
    
    def __init__(self, api_key: str = None, backup_key: str = None):
        """
        Initialize matching engine.
        
        Args:
            api_key: Primary Groq API key
            backup_key: Backup API key for rate limit fallback
        """
        self.parser = ResumeParser(api_key=api_key)
        self.scorer = ResumeScorer(api_key=api_key)
        self.backup_key = backup_key
        
        # Configurable weights (sum = 1.0)
        self.weights = {
            'skill_match': 0.50,
            'experience_depth': 0.30,
            'domain_fit': 0.20
        }
    
    def evaluate(
        self, 
        job_description: str, 
        resume_text: str,
        resume_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Complete evaluation of single resume.
        
        Args:
            job_description: The job posting
            resume_text: Raw resume text
            resume_id: Identifier for this resume
            
        Returns:
            {
                "id": str,
                "candidate_name": str,
                "final_score": float,
                "parsed_data": Dict,  # Stage 1 output
                "dimension_scores": Dict,  # Stage 2 output
                "weights": Dict,
                "score_breakdown": str,  # Natural language
                "matched_skills": List[str],
                "missing_skills": List[str],
                "processing_time_seconds": float
            }
        """
        
        start_time = time.time()
        
        print(f"📄 Evaluating {resume_id}...")
        
        # STAGE 1: Parse resume
        print(f"  ⚙️  Stage 1: Parsing resume...")
        parsed_data = self.parser.parse(resume_text)
        
        # Brief pause to avoid rate limits
        time.sleep(0.5)
        
        # STAGE 2: Score against JD
        print(f"  ⚙️  Stage 2: Scoring resume...")
        dimension_scores = self.scorer.score(job_description, parsed_data)
        
        # STAGE 3: Aggregate
        print(f"  ⚙️  Stage 3: Computing final score...")
        final_score = self._compute_final_score(dimension_scores)
        
        # Build explanation
        explanation = self._build_explanation(
            dimension_scores, 
            final_score,
            parsed_data.get("total_years_experience", 0)
        )
        
        processing_time = time.time() - start_time
        print(f"  ✅ Complete! Score: {final_score:.3f} ({processing_time:.1f}s)")
        
        return {
            "id": resume_id,
            "candidate_name": parsed_data.get("candidate_name", "Unknown"),
            "final_score": round(final_score, 3),
            "parsed_data": parsed_data,
            "dimension_scores": dimension_scores,
            "weights": self.weights,
            "score_breakdown": explanation,
            "matched_skills": dimension_scores["skill_match"]["matched_skills"],
            "missing_skills": dimension_scores["skill_match"]["missing_skills"],
            "processing_time_seconds": round(processing_time, 2)
        }
    
    def evaluate_batch(
        self,
        job_description: str,
        resumes: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple resumes.
        
        Args:
            job_description: The job posting
            resumes: List of {"id": str, "text": str}
            
        Returns:
            List of evaluation results, sorted by final_score (descending)
        """
        
        results = []
        total = len(resumes)
        
        print(f"\n🚀 Starting batch evaluation of {total} resumes...")
        print(f"⚙️  Weights: Skills={self.weights['skill_match']}, "
              f"Experience={self.weights['experience_depth']}, "
              f"Domain={self.weights['domain_fit']}\n")
        
        for i, resume in enumerate(resumes, 1):
            print(f"[{i}/{total}] ", end="")
            
            result = self.evaluate(
                job_description,
                resume["text"],
                resume["id"]
            )
            
            results.append(result)
            
            # Rate limiting: 1 second between API calls
            if i < total:
                time.sleep(1)
        
        # Sort by final_score descending
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        print(f"\n✅ Batch evaluation complete!")
        print(f"📊 Score range: {results[-1]['final_score']:.3f} - "
              f"{results[0]['final_score']:.3f}")
        
        return results
    
    def _compute_final_score(self, dimension_scores: Dict) -> float:
        """Weighted aggregation of dimension scores."""
        
        score = 0.0
        for dimension, weight in self.weights.items():
            dim_score = dimension_scores[dimension]["score"]
            score += dim_score * weight
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _build_explanation(
        self, 
        dimension_scores: Dict, 
        final_score: float,
        years_exp: float
    ) -> str:
        """Generate natural language score explanation."""
        
        lines = []
        lines.append(f"**Final Score: {final_score:.3f}**\n")
        lines.append("**Breakdown:**\n")
        
        for dimension, weight in self.weights.items():
            dim_data = dimension_scores[dimension]
            score = dim_data["score"]
            contribution = score * weight
            
            dim_name = dimension.replace('_', ' ').title()
            lines.append(f"• **{dim_name}** ({weight*100:.0f}%): {score:.2f}")
            lines.append(f"  → Contributes {contribution:.3f} to final score")
            lines.append(f"  → {dim_data['reasoning']}\n")
        
        # Add matched/missing skills
        matched = dimension_scores["skill_match"]["matched_skills"]
        missing = dimension_scores["skill_match"]["missing_skills"]
        
        lines.append(f"**Matched Skills ({len(matched)}):** {', '.join(matched) if matched else 'None'}")
        lines.append(f"**Missing Skills ({len(missing)}):** {', '.join(missing) if missing else 'None'}")
        lines.append(f"\n**Total Experience:** {years_exp:.1f} years")
        lines.append(f"\n**Overall:** {dimension_scores['overall_assessment']}")
        
        return "\n".join(lines)
    
    def update_weights(self, skill: float, experience: float, domain: float):
        """
        Update scoring weights (must sum to 1.0).
        
        Args:
            skill: Weight for skill_match (0.0-1.0)
            experience: Weight for experience_depth (0.0-1.0)
            domain: Weight for domain_fit (0.0-1.0)
        """
        total = skill + experience + domain
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0 (got {total})")
        
        self.weights = {
            'skill_match': skill,
            'experience_depth': experience,
            'domain_fit': domain
        }
        print(f"✅ Weights updated: {self.weights}")
```

---

### **4.4 Utilities (`src/utils.py`)**

```python
"""
Utility functions for data loading and metric calculation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import ndcg_score

def load_job_description(filepath: str = "data/job_descriptions/ema_ai_apps_engineer.txt") -> str:
    """Load job description from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def load_resumes(directory: str = "data/resumes/") -> List[Dict[str, str]]:
    """
    Load all resumes from directory.
    
    Returns:
        List of {"id": "resume_001", "text": "resume content"}
    """
    resumes = []
    resume_dir = Path(directory)
    
    for filepath in sorted(resume_dir.glob("*.txt")):
        resume_id = filepath.stem  # Filename without extension
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        resumes.append({
            "id": resume_id,
            "text": text
        })
    
    print(f"📁 Loaded {len(resumes)} resumes from {directory}")
    return resumes

def load_ground_truth(filepath: str = "data/ground_truth.json") -> Dict[str, float]:
    """
    Load manual labels for evaluation.
    
    Format:
    {
        "resume_001": 1.0,  // Perfect match
        "resume_002": 0.5,  // Partial match
        "resume_003": 0.0   // Poor match
    }
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def save_results(results: List[Dict], filepath: str = "results.json"):
    """Save evaluation results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to {filepath}")

def calculate_ndcg_at_k(
    predicted_scores: List[float],
    true_relevance: List[float],
    k: int = 3
) -> float:
    """
    Calculate nDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        predicted_scores: Model's scores (0.0-1.0)
        true_relevance: Ground truth relevance (0.0, 0.5, 1.0)
        k: Number of top results to consider
        
    Returns:
        nDCG score (0.0-1.0), where 1.0 is perfect ranking
    """
    # Reshape for sklearn
    true_relevance = np.array([true_relevance])
    predicted_scores = np.array([predicted_scores])
    
    return ndcg_score(true_relevance, predicted_scores, k=k)

def calculate_precision_at_k(
    predicted_scores: List[float],
    true_relevance: List[float],
    k: int = 1,
    threshold: float = 0.7
) -> float:
    """
    Calculate Precision@k.
    
    Args:
        predicted_scores: Model's scores
        true_relevance: Ground truth (0.0, 0.5, 1.0)
        k: Number of top results to consider
        threshold: Relevance threshold for "good" candidate
        
    Returns:
        Precision (0.0-1.0)
    """
    # Get top-k indices by predicted score
    top_k_indices = np.argsort(predicted_scores)[-k:][::-1]
    
    # Check if top-k are actually relevant
    relevant_count = sum(
        1 for idx in top_k_indices 
        if true_relevance[idx] >= threshold
    )
    
    return relevant_count / k

def calculate_recall_at_k(
    predicted_scores: List[float],
    true_relevance: List[float],
    k: int = 3,
    threshold: float = 0.7
) -> float:
    """
    Calculate Recall@k.
    
    Args:
        predicted_scores: Model's scores
        true_relevance: Ground truth (0.0, 0.5, 1.0)
        k: Number of top results to consider
        threshold: Relevance threshold for "good" candidate
        
    Returns:
        Recall (0.0-1.0)
    """
    # Get top-k indices
    top_k_indices = np.argsort(predicted_scores)[-k:][::-1]
    
    # Total relevant candidates
    total_relevant = sum(1 for rel in true_relevance if rel >= threshold)
    
    if total_relevant == 0:
        return 0.0
    
    # Relevant candidates in top-k
    relevant_in_top_k = sum(
        1 for idx in top_k_indices 
        if true_relevance[idx] >= threshold
    )
    
    return relevant_in_top_k / total_relevant

def print_ranking(results: List[Dict]):
    """Pretty print ranking results."""
    print("\n" + "="*80)
    print("RANKING RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n#{i} | {result['candidate_name']} ({result['id']})")
        print(f"     Score: {result['final_score']:.3f}")
        print(f"     Skills: {result['dimension_scores']['skill_match']['score']:.2f} | "
              f"Experience: {result['dimension_scores']['experience_depth']['score']:.2f} | "
              f"Domain: {result['dimension_scores']['domain_fit']['score']:.2f}")
        print(f"     Matched: {', '.join(result['matched_skills'][:5])}")
        
    print("\n" + "="*80)
```

---

## 5. Main Evaluation Script (`evaluate.py`)

```python
"""
Main evaluation script - Run full system evaluation.
"""

import os
import json
from dotenv import load_dotenv
from src.matching_engine import TwoStageMatchingEngine
from src.utils import (
    load_job_description,
    load_resumes,
    load_ground_truth,
    save_results,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    print_ranking
)

def main():
    """Run complete evaluation pipeline."""
    
    # Load environment
    load_dotenv()
    
    # Initialize engine
    engine = TwoStageMatchingEngine(
        api_key=os.getenv("GROQ_API_KEY"),
        backup_key=os.getenv("GROQ_API_KEY_2")
    )
    
    # Load data
    print("📥 Loading data...")
    job_description = load_job_description()
    resumes = load_resumes()
    ground_truth = load_ground_truth()
    
    print(f"✅ Loaded {len(resumes)} resumes")
    print(f"✅ Loaded ground truth for {len(ground_truth)} candidates\n")
    
    # Evaluate all resumes
    results = engine.evaluate_batch(job_description, resumes)
    
    # Save results
    save_results(results, "results.json")
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    predicted_scores = [r["final_score"] for r in results]
    true_relevance = [ground_truth[r["id"]] for r in results]
    
    ndcg_3 = calculate_ndcg_at_k(predicted_scores, true_relevance, k=3)
    precision_1 = calculate_precision_at_k(predicted_scores, true_relevance, k=1)
    recall_3 = calculate_recall_at_k(predicted_scores, true_relevance, k=3)
    
    print(f"\nnDCG@3:        {ndcg_3:.3f}  {'✅' if ndcg_3 >= 0.85 else '❌'} (target: ≥0.85)")
    print(f"Precision@1:   {precision_1:.3f}  {'✅' if precision_1 == 1.0 else '❌'} (target: 1.00)")
    print(f"Recall@3:      {recall_3:.3f}  {'✅' if recall_3 >= 0.9 else '❌'} (target: ≥0.90)")
    
    # Print ranking
    print_ranking(results)
    
    # Save metrics
    metrics = {
        "nDCG@3": round(ndcg_3, 3),
        "Precision@1": round(precision_1, 3),
        "Recall@3": round(recall_3, 3),
        "weights": engine.weights
    }
    
    with open("metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Evaluation complete! Results saved to results.json and metrics.json")

if __name__ == "__main__":
    main()
```

---

## 6. Ground Truth Format (`data/ground_truth.json`)

```json
{
  "resume_001": 1.0,
  "resume_002": 0.5,
  "resume_003": 0.0,
  "resume_004": 1.0,
  "resume_005": 0.5,
  "resume_006": 0.8,
  "resume_007": 0.3,
  "resume_008": 0.9,
  "resume_009": 0.4,
  "resume_010": 0.7
}
```

**Labeling Guide:**
- `1.0`: Perfect match (has 80%+ required skills, meets experience)
- `0.8-0.9`: Strong match (has most required skills, meets/exceeds experience)
- `0.5-0.7`: Partial match (has some required skills OR lacks experience)
- `0.2-0.4`: Weak match (few required skills, wrong domain)
- `0.0`: Poor match (wrong field entirely)

---

## 7. Streamlit Dashboard (`app.py`)

```python
"""
Streamlit dashboard for interactive resume evaluation.
"""

import streamlit as st
import json
import plotly.graph_objects as go
from pathlib import Path
import os
from dotenv import load_dotenv
from src.matching_engine import TwoStageMatchingEngine
from src.utils import load_job_description, load_resumes

# Page config
st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide"
)

# Load environment
load_dotenv()

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Title
st.title("🎯 Two-Stage LLM Resume Matcher")
st.markdown("**Intelligent resume evaluation with explainable AI**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Weight sliders
    st.subheader("Scoring Weights")
    skill_weight = st.slider("Skill Match", 0.0, 1.0, 0.50, 0.05)
    exp_weight = st.slider("Experience Depth", 0.0, 1.0, 0.30, 0.05)
    domain_weight = st.slider("Domain Fit", 0.0, 1.0, 0.20, 0.05)
    
    total_weight = skill_weight + exp_weight + domain_weight
    st.metric("Total Weight", f"{total_weight:.2f}", 
              delta="Valid" if abs(total_weight - 1.0) < 0.01 else "Must = 1.0")
    
    st.divider()
    
    # Evaluation button
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        if abs(total_weight - 1.0) > 0.01:
            st.error("Weights must sum to 1.0!")
        else:
            with st.spinner("Evaluating resumes..."):
                # Initialize engine
                engine = TwoStageMatchingEngine()
                engine.update_weights(skill_weight, exp_weight, domain_weight)
                
                # Load data
                jd = load_job_description()
                resumes = load_resumes()
                
                # Evaluate
                results = engine.evaluate_batch(jd, resumes)
                st.session_state.results = results
                
                st.success("✅ Evaluation complete!")

# Main area
if st.session_state.results is None:
    st.info("👈 Click 'Run Evaluation' to start")
    
    # Show sample JD
    with st.expander("📋 Job Description Preview"):
        jd = load_job_description()
        st.text(jd[:500] + "...")
else:
    results = st.session_state.results
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Candidates", len(results))
    
    with col2:
        top_score = results[0]['final_score']
        st.metric("Top Score", f"{top_score:.3f}")
    
    with col3:
        avg_score = sum(r['final_score'] for r in results) / len(results)
        st.metric("Average Score", f"{avg_score:.3f}")
    
    with col4:
        avg_time = sum(r['processing_time_seconds'] for r in results) / len(results)
        st.metric("Avg Processing Time", f"{avg_time:.1f}s")
    
    st.divider()
    
    # Score distribution chart
    st.subheader("📊 Score Distribution")
    
    fig = go.Figure(data=[
        go.Bar(
            x=[r['candidate_name'] for r in results],
            y=[r['final_score'] for r in results],
            marker_color=['#00cc88' if r['final_score'] >= 0.7 else 
                         '#ffa500' if r['final_score'] >= 0.5 else '#ff4444' 
                         for r in results]
        )
    ])
    fig.update_layout(
        xaxis_title="Candidate",
        yaxis_title="Final Score",
        yaxis_range=[0, 1.0],
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Candidate details
    st.subheader("👥 Candidate Rankings")
    
    for i, result in enumerate(results, 1):
        with st.expander(f"#{i} | {result['candidate_name']} - Score: {result['final_score']:.3f}"):
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📝 Breakdown", "🔍 Details", "📄 Raw Data"])
            
            with tab1:
                st.markdown(result['score_breakdown'])
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Dimension Scores:**")
                    dims = result['dimension_scores']
                    st.metric("Skill Match", f"{dims['skill_match']['score']:.2f}")
                    st.metric("Experience Depth", f"{dims['experience_depth']['score']:.2f}")
                    st.metric("Domain Fit", f"{dims['domain_fit']['score']:.2f}")
                
                with col2:
                    st.markdown("**Parsed Data:**")
                    parsed = result['parsed_data']
                    st.metric("Total Experience", f"{parsed['total_years_experience']:.1f} years")
                    st.metric("Skills Extracted", len(parsed['skills']))
                    st.metric("Positions", len(parsed['experience']))
            
            with tab3:
                st.json(result)
    
    # Download button
    st.divider()
    st.download_button(
        label="⬇️ Download Results (JSON)",
        data=json.dumps(results, indent=2),
        file_name="evaluation_results.json",
        mime="application/json"
    )
```

---

## 8. Testing Framework (`tests/test_engine.py`)

```python
"""
Unit tests for matching engine.
"""

import pytest
from src.matching_engine import TwoStageMatchingEngine

def test_weight_validation():
    """Test weight validation."""
    engine = TwoStageMatchingEngine()
    
    # Valid weights
    engine.update_weights(0.5, 0.3, 0.2)
    assert engine.weights['skill_match'] == 0.5
    
    # Invalid weights (don't sum to 1.0)
    with pytest.raises(ValueError):
        engine.update_weights(0.5, 0.5, 0.5)

def test_score_clamping():
    """Test scores are clamped to [0, 1]."""
    engine = TwoStageMatchingEngine()
    
    # Mock dimension scores
    dim_scores = {
        'skill_match': {'score': 1.5},  # Over 1.0
        'experience_depth': {'score': 0.5},
        'domain_fit': {'score': 0.5}
    }
    
    final = engine._compute_final_score(dim_scores)
    assert 0.0 <= final <= 1.0

def test_batch_evaluation():
    """Test batch processing."""
    engine = TwoStageMatchingEngine()
    
    jd = "Looking for Python developer with 3+ years experience"
    resumes = [
        {"id": "test_1", "text": "Python developer with 5 years experience"},
        {"id": "test_2", "text": "Java developer with 2 years experience"}
    ]
    
    results = engine.evaluate_batch(jd, resumes)
    
    assert len(results) == 2
    assert results[0]['final_score'] >= results[1]['final_score']  # Sorted
```

---

## 9. `.gitignore`

```
# Environment
.env
*.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Results
results.json
metrics.json
*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## 10. Quick Start Commands

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your Groq API keys

# 3. Run evaluation
python evaluate.py

# 4. Launch dashboard
streamlit run app.py

# 5. Run tests
pytest tests/
```

---

## 11. Success Criteria Checklist

✅ **Core Functionality:**
- [ ] Parser extracts structured data from resumes
- [ ] Scorer evaluates on 3 dimensions
- [ ] Engine aggregates with configurable weights
- [ ] Batch processing works for 10+ resumes

✅ **Evaluation:**
- [ ] nDCG@3 ≥ 0.85
- [ ] Precision@1 = 1.0
- [ ] All metrics saved to JSON

✅ **Documentation:**
- [ ] README explains architecture decisions
- [ ] Comments in all functions
- [ ] Examples in docstrings

✅ **Reproducibility:**
- [ ] requirements.txt complete
- [ ] .env.example provided
- [ ] Clear setup instructions

---

## 12. Time Estimates

| Task | Time | Priority |
|------|------|----------|
| Setup + dependencies | 30 min | Critical |
| Implement parser | 3 hours | Critical |
| Implement scorer | 3 hours | Critical |
| Implement engine | 2 hours | Critical |
| Create test resumes + labels | 2 hours | Critical |
| Run evaluation | 1 hour | Critical |
| Write README | 3 hours | Critical |
| Build Streamlit dashboard | 4 hours | High |
| Write tests | 2 hours | Medium |
| **TOTAL** | **20 hours** | |

---

**This spec is complete and ready to implement. Start with the parser, test it on 1-2 resumes, then build scorer, then engine. Good luck! 🚀**