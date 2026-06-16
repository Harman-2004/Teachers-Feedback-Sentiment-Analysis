"""
nlp/sentiment.py
----------------
Sentiment analysis module for teacher feedback.

Provides:
  • Fine-grained 3-class sentiment classification (Positive / Neutral / Negative)
  • Mixed-sentiment detection — returns all class probabilities, not just the top label
  • A numeric polarity score in [0.0, 10.0] used by the scoring engine
  • Batch processing with deduplication support

Model: cardiffnlp/twitter-roberta-base-sentiment-latest  (3-class RoBERTa)
Fallback: distilbert-base-uncased-finetuned-sst-2-english (2-class DistilBERT)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ── Model identifiers ────────────────────────────────────────────────────────
PRIMARY_MODEL  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Normalise whatever label string the model emits → canonical form
_LABEL_NORM: Dict[str, str] = {
    "positive": "Positive", "POSITIVE": "Positive", "LABEL_2": "Positive",
    "negative": "Negative", "NEGATIVE": "Negative", "LABEL_0": "Negative",
    "neutral":  "Neutral",  "NEUTRAL":  "Neutral",  "LABEL_1": "Neutral",
}

# Mixed-sentiment threshold: if the gap between top-1 and top-2 scores is below
# this value the text is flagged as mixed.
MIXED_THRESHOLD: float = 0.20


# ── Private helpers ──────────────────────────────────────────────────────────
@lru_cache(maxsize=2)
def _load_pipeline(model_name: str):
    """Load and *cache* a HuggingFace sentiment pipeline (once per session)."""
    import os
    if os.environ.get("FORCE_RULE_BASED") == "1":
        logger.info("Force rule-based mode is active. Skipping PyTorch/Transformers load.")
        return None, "fallback"
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        try:
            logger.info("Loading sentiment model: %s", model_name)
            pipe = hf_pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device,
                truncation=True,
                max_length=512,
                # Return ALL class scores so we can detect mixed sentiment
                top_k=None,
                return_all_scores=True,
            )
            logger.info("Sentiment model ready.")
            return pipe, "multi"
        except Exception as exc:
            logger.warning("Primary model failed (%s). Using fallback.", exc)
            pipe = hf_pipeline(
                "sentiment-analysis",
                model=FALLBACK_MODEL,
                device=device,
                truncation=True,
                max_length=512,
                return_all_scores=True,
            )
            return pipe, "multi"
    except (ImportError, OSError, Exception) as exc:
        logger.error("Failed to load sentiment pipeline: %s. Using rule-based fallback.", exc)
        return None, "fallback"


def _normalise_scores(raw: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Convert raw model output list → {Positive, Neutral, Negative} float map.
    Handles both 2-class and 3-class model outputs.
    """
    mapped: Dict[str, float] = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    for item in raw:
        canonical = _LABEL_NORM.get(item["label"], item["label"].capitalize())
        mapped[canonical] = mapped.get(canonical, 0.0) + float(item["score"])

    # For 2-class models: distribute remaining probability to Neutral
    total = sum(mapped.values())
    if total < 0.99:
        mapped["Neutral"] += 1.0 - total

    return mapped


def _polarity_score(probs: Dict[str, float]) -> float:
    """
    Convert class probabilities → a single polarity float in [0.0, 10.0].

    Formula: (P(Pos) - P(Neg) + 1) / 2 × 10
    → 10.0 = 100 % positive confidence
    → 5.0  = equal positive / negative (neutral)
    → 0.0  = 100 % negative confidence
    """
    pos = probs.get("Positive", 0.0)
    neg = probs.get("Negative", 0.0)
    raw = (pos - neg + 1.0) / 2.0       # → [0, 1]
    return round(raw * 10.0, 4)


def _detect_mixed(probs: Dict[str, float]) -> bool:
    """Return True when the two highest-scoring classes are close."""
    sorted_vals = sorted(probs.values(), reverse=True)
    if len(sorted_vals) < 2:
        return False
    return (sorted_vals[0] - sorted_vals[1]) < MIXED_THRESHOLD


# ── Public API ───────────────────────────────────────────────────────────────
def analyze_sentiment(
    texts: List[str],
    model_name: str = PRIMARY_MODEL,
    batch_size: int = 16,
    deduplicate: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run sentiment analysis on a list of feedback strings.

    Parameters
    ----------
    texts       : raw feedback strings
    model_name  : HuggingFace model ID (default: RoBERTa 3-class)
    batch_size  : texts per inference batch
    deduplicate : skip exact duplicates and re-use cached results

    Returns
    -------
    list[dict]  — one dict per *input* text (order preserved), containing:
        text          str   original input
        label         str   "Positive" | "Neutral" | "Negative"
        probs         dict  {Positive, Neutral, Negative} → float
        polarity      float 0.0 – 10.0  (0 = fully negative, 10 = fully positive)
        confidence    float model confidence in the top label  (0 – 1)
        is_mixed      bool  True when sentiment is ambiguous
    """
    if not texts:
        return []

    pipe, mode = _load_pipeline(model_name)
    if pipe is None:
        return _analyze_sentiment_rule_based(texts)

    # ── Deduplication ────────────────────────────────────────────────────────
    cache: Dict[str, Dict[str, Any]] = {}
    unique_texts: List[str] = []
    if deduplicate:
        for t in texts:
            if t not in cache:
                cache[t] = None          # placeholder
                unique_texts.append(t)
    else:
        unique_texts = list(texts)

    # ── Batched inference ────────────────────────────────────────────────────
    unique_results: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(unique_texts), batch_size):
        batch = unique_texts[i : i + batch_size]
        try:
            raw_batch = pipe(batch)
        except Exception as exc:
            logger.error("Batch %d failed: %s", i // batch_size, exc)
            raw_batch = [[{"label": "neutral", "score": 0.5}]] * len(batch)

        for text, raw in zip(batch, raw_batch):
            # raw is a list of {label, score} dicts (all classes)
            if isinstance(raw, dict):
                raw = [raw]
            probs = _normalise_scores(raw)
            top_label = max(probs, key=lambda k: probs[k])
            unique_results[text] = {
                "text":       text,
                "label":      top_label,
                "probs":      {k: round(v, 4) for k, v in probs.items()},
                "polarity":   _polarity_score(probs),
                "confidence": round(probs[top_label], 4),
                "is_mixed":   _detect_mixed(probs),
            }

    # ── Re-expand to original order ──────────────────────────────────────────
    return [unique_results[t] for t in texts]


def sentiment_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate a list of sentiment results into dataset-level statistics.

    Returns
    -------
    dict
        counts        {Positive, Neutral, Negative} → int
        percentages   {Positive, Neutral, Negative} → float %
        avg_polarity  mean polarity (0–10)
        avg_confidence mean model confidence
        mixed_count   number of mixed-sentiment texts
        overall       dominant label
    """
    counts    = {"Positive": 0, "Neutral": 0, "Negative": 0}
    pol_sum   = 0.0
    conf_sum  = 0.0
    mixed_cnt = 0

    for r in results:
        lbl = r.get("label", "Neutral")
        counts[lbl] = counts.get(lbl, 0) + 1
        pol_sum  += r.get("polarity", 5.0)
        conf_sum += r.get("confidence", 0.5)
        if r.get("is_mixed", False):
            mixed_cnt += 1

    n = len(results) if results else 1
    return {
        "counts":          counts,
        "percentages":     {k: round(v / n * 100, 2) for k, v in counts.items()},
        "avg_polarity":    round(pol_sum  / n, 4),
        "avg_confidence":  round(conf_sum / n, 4),
        "mixed_count":     mixed_cnt,
        "overall":         max(counts, key=lambda k: counts[k]),
    }


def _analyze_sentiment_rule_based(texts: List[str]) -> List[Dict[str, Any]]:
    """Lightweight but advanced rule-based sentiment analyzer fallback for low-memory environments."""
    results = []
    
    # Expanded lexicons
    pos_words = {
        "great", "excellent", "best", "good", "amazing", "wonderful", "love", "liked", "engaging", 
        "helpful", "supportive", "patience", "clear", "clarity", "structured", "professional", 
        "punctual", "enthusiastic", "passionate", "interactive", "support", "fair", "real-world",
        "approachable", "encouraging", "informative", "inspiring", "organized", "knowledgeable",
        "patient", "friendly", "kind", "brilliant", "outstanding", "accessible",
        "concise", "articulate", "fairly", "prompt", "effectively", "recommend"
    }
    
    neg_words = {
        "difficult", "bad", "worst", "unclear", "fast", "slow", "boring", "late", "delayed", 
        "unresponsive", "outdated", "monotonous", "poor", "chaotic", "unproductive", "unapproachable", 
        "dismissive", "workload", "too many", "inconsistent", "vague", "disorganized", "unorganized",
        "confused", "dry", "rude", "harsh", "unfair", "frustrated", "frustrating", "mumble",
        "mumbling", "terrible", "waste", "useless", "ignore", "ignored", "ignores", "disappointed"
    }
    
    negations = {"not", "no", "never", "neither", "nor", "hardly", "scarcely", "barely", "dont", "doesnt", "didnt", "wasnt", "werent", "havent", "hadnt", "couldnt", "wouldnt", "shouldnt", "cant", "cannot", "without"}
    intensifiers = {"very", "extremely", "really", "incredibly", "highly", "absolutely", "so", "super", "truly"}
    
    import re
    
    for text in texts:
        lower = text.lower()
        # Clean text punctuation for easier word matching
        words = re.findall(r"\b[a-z']+\b", lower)
        
        pos_score = 0.0
        neg_score = 0.0
        
        for idx, word in enumerate(words):
            # Check if word is in sentiment lexicon
            is_pos = word in pos_words
            is_neg = word in neg_words
            
            if not is_pos and not is_neg:
                continue
                
            # Check negation window of 2 words before
            negated = False
            intensity = 1.0
            
            start_window = max(0, idx - 2)
            for j in range(start_window, idx):
                prev_word = words[j]
                if prev_word in negations:
                    negated = True
                elif prev_word in intensifiers:
                    intensity = 1.5
            
            if is_pos:
                if negated:
                    neg_score += 1.0 * intensity
                else:
                    pos_score += 1.0 * intensity
            elif is_neg:
                if negated:
                    pos_score += 1.0 * intensity
                else:
                    neg_score += 1.0 * intensity
        
        total = pos_score + neg_score
        
        if total == 0:
            probs = {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33}
            top_label = "Neutral"
        else:
            pos_p = pos_score / total
            neg_p = neg_score / total
            
            # Map probabilities
            probs = {
                "Positive": round(pos_p * 0.8, 4),
                "Neutral": 0.20,
                "Negative": round(neg_p * 0.8, 4)
            }
            
            # Re-normalize
            sum_p = sum(probs.values())
            probs = {k: round(v / sum_p, 4) for k, v in probs.items()}
            
            # Determine top label
            # If the difference between positive and negative is very small, make it Neutral
            diff = abs(probs["Positive"] - probs["Negative"])
            if diff < 0.15:
                top_label = "Neutral"
            elif probs["Positive"] > probs["Negative"]:
                top_label = "Positive"
            else:
                top_label = "Negative"
                
        pos = probs.get("Positive", 0.0)
        neg = probs.get("Negative", 0.0)
        pol = round(((pos - neg + 1.0) / 2.0) * 10.0, 4)
        
        # Mixed check
        is_mixed = bool(pos_score > 0 and neg_score > 0 and abs(pos_score - neg_score) <= 2)
        
        results.append({
            "text": text,
            "label": top_label,
            "probs": probs,
            "polarity": pol,
            "confidence": round(probs[top_label], 4),
            "is_mixed": is_mixed,
        })
    return results
