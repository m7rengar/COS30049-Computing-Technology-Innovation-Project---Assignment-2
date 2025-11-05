"""
Text preprocessing utilities for spam detection
Adjust these functions to match your training preprocessing pipeline
"""

import re
from typing import List

class TextPreprocessor:
    def __init__(self, vectorizer_path=None):
        """
        Initialize preprocessor
        
        Args:
            vectorizer_path: Path to saved vectorizer (TfidfVectorizer, CountVectorizer, etc.)
        """
        self.vectorizer = None
        if vectorizer_path:
            import joblib
            self.vectorizer = joblib.load(vectorizer_path)
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        Customize based on your training preprocessing
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (optional - adjust as needed)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess(self, text: str):
        """
        Full preprocessing pipeline
        Returns vectorized text ready for model prediction
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # If you have a vectorizer (TF-IDF, etc.), use it
        if self.vectorizer:
            return self.vectorizer.transform([cleaned])
        
        # Otherwise return cleaned text
        return [cleaned]
    
    def preprocess_batch(self, texts: List[str]):
        """Preprocess multiple texts at once"""
        cleaned = [self.clean_text(t) for t in texts]
        
        if self.vectorizer:
            return self.vectorizer.transform(cleaned)
        
        return cleaned