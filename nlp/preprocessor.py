"""
nlp/preprocessor.py
-------------------
Text preprocessing pipeline for teacher feedback sentiment analysis.
Handles lowercasing, URL removal, punctuation cleaning, tokenization,
negation preservation, and lemmatization.
"""

from __future__ import annotations
import re
from typing import List, Union

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK datasets are downloaded silently if available
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    # Fallback to pure Python stopwords set if NLTK data download fails
    _STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very",
        "s", "t", "can", "will", "just", "don", "should", "now"
    }
    _LEMMATIZER = None

# Preserve negation words so sentiment context is not lost during pre-processing
_NEGATION_WORDS = {
    "no", "not", "nor", "neither", "never", "cannot", "cant", "dont", "doesnt",
    "didnt", "wasnt", "werent", "havent", "hadnt", "couldnt", "wouldnt", "shouldnt"
}

_CUSTOM_STOPWORDS = _STOPWORDS - _NEGATION_WORDS

# Regex patterns
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s']")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess a single text entry.
    Preserves negation words to retain critical sentiment signals.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercasing
    cleaned = text.lower()

    # 2. Remove URLs
    cleaned = _URL_PATTERN.sub(" ", cleaned)

    # 3. Handle contractions/negations format (e.g. "don't" -> "dont")
    cleaned = cleaned.replace("n't", "nt").replace("'t", "t")

    # 4. Remove unnecessary punctuation
    cleaned = _PUNCTUATION_PATTERN.sub(" ", cleaned)

    # 5. Tokenization & Extra whitespace handling
    tokens = [t.strip() for t in _WHITESPACE_PATTERN.split(cleaned) if t.strip()]

    # 6. Stopwords handling & Lemmatization while preserving negations
    processed_tokens = []
    for token in tokens:
        if token not in _CUSTOM_STOPWORDS:
            if _LEMMATIZER is not None:
                try:
                    token = _LEMMATIZER.lemmatize(token)
                except Exception:
                    pass
            processed_tokens.append(token)

    return " ".join(processed_tokens)


def preprocess_documents(texts: Union[List[str], pd.Series]) -> List[str]:
    """Preprocess a list or series of text documents."""
    return [preprocess_text(t) for t in texts]
