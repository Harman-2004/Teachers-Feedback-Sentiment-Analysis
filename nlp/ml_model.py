"""
nlp/ml_model.py
---------------
Reusable container classes for Word2Vec document vectorization
and ML model pipeline serialization.
"""

from __future__ import annotations
from typing import List, Any
import numpy as np
from gensim.models import Word2Vec
from nlp.preprocessor import preprocess_text


class Word2VecVectorizer:
    """Helper class to train Word2Vec embeddings and convert documents into mean vectors."""
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, seed: int = 42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.seed = seed
        self.model: Word2Vec | None = None

    def fit(self, tokenized_docs: List[List[str]]):
        self.model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            seed=self.seed
        )
        return self

    def transform(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        vectors = []
        for doc in tokenized_docs:
            valid_words = [w for w in doc if self.model and w in self.model.wv]
            if valid_words:
                vec = np.mean([self.model.wv[w] for w in valid_words], axis=0)
            else:
                vec = np.zeros(self.vector_size)
            vectors.append(vec)
        return np.array(vectors)

    def fit_transform(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        self.fit(tokenized_docs)
        return self.transform(tokenized_docs)


class MLPipeline:
    """Unified wrapper pipeline for pre-processing, vectorization, and inference."""
    def __init__(self, model_type: str, vectorizer: Any, classifier: Any):
        self.model_type = model_type
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict(self, texts: List[str]) -> List[str]:
        cleaned_texts = [preprocess_text(t) for t in texts]
        if self.model_type == "word2vec":
            tokenized = [t.split() for t in cleaned_texts]
            features = self.vectorizer.transform(tokenized)
        else:
            features = self.vectorizer.transform(cleaned_texts)
        return self.classifier.predict(features).tolist()
