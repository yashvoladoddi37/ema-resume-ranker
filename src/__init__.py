# Two-Stage Resume Matching Engine
"""
A production-grade resume matching system using a two-stage LLM pipeline.

Stage 1 (Parser): Extracts structured data from raw resume text
Stage 2 (Scorer): Scores parsed data against job description on multiple dimensions

Usage:
    from src.matching_engine import TwoStageMatchingEngine
    
    engine = TwoStageMatchingEngine()
    result = engine.evaluate(job_description, resume_text, resume_id)
"""

from .resume_parser import ResumeParser
from .resume_scorer import ResumeScorer
from .matching_engine import TwoStageMatchingEngine
from .utils import (
    load_job_description,
    load_resumes,
    load_ground_truth,
    save_results,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k
)

__all__ = [
    "ResumeParser",
    "ResumeScorer", 
    "TwoStageMatchingEngine",
    "load_job_description",
    "load_resumes",
    "load_ground_truth",
    "save_results",
    "calculate_ndcg_at_k",
    "calculate_precision_at_k",
    "calculate_recall_at_k"
]
