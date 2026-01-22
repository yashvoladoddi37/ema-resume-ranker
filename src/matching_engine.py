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
            "weights": self.weights.copy(),
            "score_breakdown": explanation,
            "matched_skills": dimension_scores["skill_match"].get("matched_skills", []),
            "missing_skills": dimension_scores["skill_match"].get("missing_skills", []),
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
            dim_data = dimension_scores.get(dimension, {})
            dim_score = dim_data.get("score", 0.5)
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
            dim_data = dimension_scores.get(dimension, {})
            score = dim_data.get("score", 0.5)
            contribution = score * weight
            
            dim_name = dimension.replace('_', ' ').title()
            lines.append(f"• **{dim_name}** ({weight*100:.0f}%): {score:.2f}")
            lines.append(f"  → Contributes {contribution:.3f} to final score")
            lines.append(f"  → {dim_data.get('reasoning', 'No reasoning provided')}\n")
        
        # Add matched/missing skills
        skill_data = dimension_scores.get("skill_match", {})
        matched = skill_data.get("matched_skills", [])
        missing = skill_data.get("missing_skills", [])
        
        lines.append(f"**Matched Skills ({len(matched)}):** {', '.join(matched) if matched else 'None'}")
        lines.append(f"**Missing Skills ({len(missing)}):** {', '.join(missing) if missing else 'None'}")
        lines.append(f"\n**Total Experience:** {years_exp:.1f} years")
        lines.append(f"\n**Overall:** {dimension_scores.get('overall_assessment', 'N/A')}")
        
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
