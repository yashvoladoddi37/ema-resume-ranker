import re
import unicodedata
from typing import Set

class TextPreprocessor:
    """
    Handles text cleaning and normalization for resumes and job descriptions.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalize text for consistent scoring.
        """
        if not text:
            return ""
        
        # 1. Unicode normalization (NFKC ensures consistency)
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Convert to lowercase
        text = text.lower()
        
        # 3. Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 4. Remove obvious metadata (emails/urls) - optional but useful for LLMA privacy
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'http\S+|www\S+', '[URL]', text)
        
        return text

    @staticmethod
    def extract_keywords(text: str, taxonomy: Set[str]) -> Set[str]:
        """
        Extract pre-defined skills from text using simple substring matching.
        Note: More advanced NER can be added later, but this is a stable baseline.
        """
        found = set()
        text_lower = text.lower()
        for skill in taxonomy:
            if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                found.add(skill)
        return found
