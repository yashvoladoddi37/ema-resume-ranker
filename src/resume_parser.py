"""
Resume Parser - Stage 1 of Two-Stage Pipeline
Extracts structured data from raw resume text using LLM.
"""

from groq import Groq
import json
import os
from typing import Dict, Any, List
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
        try:
            data["total_years_experience"] = float(data["total_years_experience"])
        except (TypeError, ValueError):
            data["total_years_experience"] = 0.0
        
        # Ensure skills is a list
        if not isinstance(data.get("skills"), list):
            data["skills"] = []
        
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
            "domains": [],
            "parse_error": True
        }
