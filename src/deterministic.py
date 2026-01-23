import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

@dataclass
class ExperienceProfile:
    """Deterministic experience extraction"""
    total_years: float
    roles: List[str]
    companies: List[str]
    
@dataclass
class SkillProfile:
    """Deterministic skill matching"""
    matched_required: Set[str]
    matched_preferred: Set[str]
    missing_required: Set[str]
    skill_coverage_score: float  # 0.0 to 1.0

class DeterministicExtractor:
    """
    Production-grade rule-based extractors for verifiable metrics.
    Provides HITL-friendly, auditable scoring components.
    """
    
    # Ema-specific skill taxonomy
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
    
    @staticmethod
    def extract_years_of_experience(text: str) -> float:
        """
        Extract total years of experience using multiple heuristics.
        Returns: float (years)
        """
        text_lower = text.lower()
        years_from_mention = 0.0
        years_from_ranges = 0.0
        
        # Heuristic 1: Explicit mentions ("X years")
        year_patterns = [
            r'(\d+)\+?\s*years?',
            r'(\d+)\s*yrs?',
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                years_from_mention = max(years_from_mention, max(int(m) for m in matches))
        
        # Heuristic 2: Date ranges (2020 - 2025)
        # Exclude education if possible by ignoring ranges before 2010 (crude but effective for this context)
        date_ranges = re.findall(r'(\d{4})\s*[-–]\s*(\d{4}|present)', text_lower)
        for start, end in date_ranges:
            start_year = int(start)
            if start_year < 2010: continue # Likely education or too old for this JD's relevance
            
            end_year = 2026 if end == 'present' else int(end)
            years_from_ranges += (end_year - start_year)
        
        # Take the more conservative estimate or the one that's non-zero
        total_years = max(years_from_mention, years_from_ranges)
        
        return round(total_years, 1)
    
    @staticmethod
    def extract_skills(text: str) -> SkillProfile:
        """
        Rule-based skill extraction with exact matching.
        Returns: SkillProfile with coverage metrics
        """
        text_lower = text.lower()
        
        # Find matches
        matched_required = {
            skill for skill in DeterministicExtractor.REQUIRED_SKILLS
            if re.search(rf'\b{re.escape(skill)}\b', text_lower)
        }
        
        matched_preferred = {
            skill for skill in DeterministicExtractor.PREFERRED_SKILLS
            if re.search(rf'\b{re.escape(skill)}\b', text_lower)
        }
        
        missing_required = DeterministicExtractor.REQUIRED_SKILLS - matched_required
        
        # Calculate coverage score
        required_coverage = len(matched_required) / len(DeterministicExtractor.REQUIRED_SKILLS)
        preferred_coverage = len(matched_preferred) / len(DeterministicExtractor.PREFERRED_SKILLS)
        
        # Weighted: 70% required, 30% preferred
        skill_coverage_score = (0.7 * required_coverage) + (0.3 * preferred_coverage)
        
        return SkillProfile(
            matched_required=matched_required,
            matched_preferred=matched_preferred,
            missing_required=missing_required,
            skill_coverage_score=round(skill_coverage_score, 3)
        )
    
    @staticmethod
    def calculate_domain_relevance(text: str) -> Dict[str, float]:
        """
        Calculate domain-specific keyword density.
        Returns: Dict with AI and Support relevance scores (0.0 to 1.0)
        """
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return {'ai_relevance': 0.0, 'support_relevance': 0.0}
        
        # Count keyword occurrences
        ai_count = sum(
            1 for word in words
            if any(kw in word for kw in DeterministicExtractor.AI_KEYWORDS)
        )
        
        support_count = sum(
            1 for word in words
            if any(kw in word for kw in DeterministicExtractor.SUPPORT_KEYWORDS)
        )
        
        # Normalize by text length (cap at 10% for sanity)
        ai_relevance = min(ai_count / total_words * 10, 1.0)
        support_relevance = min(support_count / total_words * 10, 1.0)
        
        return {
            'ai_relevance': round(ai_relevance, 3),
            'support_relevance': round(support_relevance, 3)
        }
    
    @staticmethod
    def calculate_deterministic_score(
        years_exp: float,
        skill_profile: SkillProfile,
        domain_relevance: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Combine all deterministic signals into a final score.
        
        Scoring breakdown:
        - Experience (20%): 3+ years = 1.0, linear scale below
        - Skill Coverage (40%): Based on required/preferred match
        - AI Relevance (20%): Keyword density
        - Support Relevance (20%): Keyword density
        
        Returns: (final_score, breakdown_dict)
        """
        # Experience score (3+ years = full points)
        exp_score = min(years_exp / 3.0, 1.0)
        
        # Component scores
        skill_score = skill_profile.skill_coverage_score
        ai_score = domain_relevance['ai_relevance']
        support_score = domain_relevance['support_relevance']
        
        # Weighted combination
        final_score = (
            0.20 * exp_score +
            0.40 * skill_score +
            0.20 * ai_score +
            0.20 * support_score
        )
        
        breakdown = {
            'experience_score': round(exp_score, 3),
            'skill_coverage_score': round(skill_score, 3),
            'ai_relevance_score': round(ai_score, 3),
            'support_relevance_score': round(support_score, 3),
            'deterministic_final': round(final_score, 3)
        }
        
        return round(final_score, 3), breakdown
