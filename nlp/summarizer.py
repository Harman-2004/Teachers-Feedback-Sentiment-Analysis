"""
nlp/summarizer.py
-----------------
Summarization & keyword-extraction module for teacher feedback.

Provides:
  • summarize_feedback()        — abstractive paragraph summary (BART / DistilBART)
  • summarize_by_sentiment()    — separate summaries per sentiment label
  • extract_strengths()         — returns concise strength phrases (rule + extraction)
  • extract_improvements()      — returns concise improvement phrases
  • extract_keywords()          — frequency-ranked keywords
  • generate_executive_summary() — plain-text executive report block

Model strategy
--------------
  Primary   : facebook/bart-large-cnn  (best quality, ~1.6 GB)
  Fallback  : sshleifer/distilbart-cnn-12-6  (~300 MB, faster)
  Nano mode : extractive-only (no download needed, instant, lower quality)

The pipeline auto-selects based on what is importable and available.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from functools import lru_cache
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

PRIMARY_MODEL  = "facebook/bart-large-cnn"
FALLBACK_MODEL = "sshleifer/distilbart-cnn-12-6"
MIN_LEN = 25
MAX_LEN = 130


# ── Model loader ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_summarizer(model_name: str):
    """Load and cache a HuggingFace summarization pipeline."""
    import os
    if os.environ.get("FORCE_RULE_BASED") == "1":
        logger.info("Force rule-based mode is active. Skipping Summarizer load.")
        return None
    import torch
    from transformers import pipeline as hf_pipeline
    device = 0 if torch.cuda.is_available() else -1
    try:
        logger.info("Loading summarizer: %s", model_name)
        pipe = hf_pipeline(
            "summarization",
            model=model_name,
            device=device,
            truncation=True,
        )
        logger.info("Summarizer ready.")
        return pipe
    except Exception as exc:
        logger.warning("Summarizer %s failed (%s). Trying fallback.", model_name, exc)
        try:
            pipe = hf_pipeline(
                "summarization",
                model=FALLBACK_MODEL,
                device=device,
                truncation=True,
            )
            return pipe
        except Exception as exc2:
            logger.error("Both summarizers failed: %s. Using extractive fallback.", exc2)
            return None


# ── Extractive fallback ───────────────────────────────────────────────────────
def _extractive_summary(text: str, n_sentences: int = 3) -> str:
    """Return the first *n_sentences* as a lightweight extractive summary."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chosen = [s.strip() for s in sentences if len(s.strip()) > 20][:n_sentences]
    return " ".join(chosen) if chosen else text[:300]


# ── Public API ────────────────────────────────────────────────────────────────
def summarize_feedback(
    texts: List[str],
    model_name: str = PRIMARY_MODEL,
    min_length: int = MIN_LEN,
    max_length: int = MAX_LEN,
    max_input_chars: int = 3500,
) -> str:
    """
    Generate an abstractive summary of one or more feedback strings.

    Parameters
    ----------
    texts           : list of raw feedback strings (concatenated before summarising)
    model_name      : HuggingFace model ID
    min_length      : minimum summary token length
    max_length      : maximum summary token length
    max_input_chars : hard truncation before model inference

    Returns
    -------
    str — summary paragraph
    """
    if not texts:
        return "No feedback available for summarisation."

    combined = " ".join(t.strip() for t in texts if t.strip())
    combined = combined[:max_input_chars]

    if len(combined.split()) < 25:
        return combined  # Too short to meaningfully summarise

    pipe = _load_summarizer(model_name)
    if pipe is None:
        return _extractive_summary(combined)

    try:
        result = pipe(
            combined,
            min_length=min_length,
            max_length=max_length,
            do_sample=False,
        )
        return result[0]["summary_text"].strip()
    except Exception as exc:
        logger.error("Summarisation failed: %s. Using extractive.", exc)
        return _extractive_summary(combined)


def summarize_by_sentiment(
    sentiment_results: List[Dict[str, Any]],
    model_name: str = PRIMARY_MODEL,
) -> Dict[str, str]:
    """
    Generate separate summaries for Positive, Neutral and Negative feedback subsets.

    Parameters
    ----------
    sentiment_results : list[dict] from ``sentiment.analyze_sentiment``

    Returns
    -------
    dict  {label: summary_str}
    """
    grouped: Dict[str, List[str]] = {"Positive": [], "Neutral": [], "Negative": []}
    for r in sentiment_results:
        lbl = r.get("label", "Neutral")
        grouped[lbl].append(r.get("text", ""))

    return {
        lbl: (
            summarize_feedback(txts, model_name=model_name)
            if txts
            else f"No {lbl.lower()} feedback found."
        )
        for lbl, txts in grouped.items()
    }


# ── Strength & improvement phrase extraction ──────────────────────────────────
# Positive aspect phrases mapped to human-readable strength labels
_STRENGTH_MAP: Dict[str, str] = {
    "clear": "clear explanations",
    "clarity": "clear explanations",
    "explains well": "clear explanations",
    "articulate": "articulate communication",
    "well-structured": "well-structured lectures",
    "engaging": "highly engaging teaching style",
    "interactive": "interactive classroom sessions",
    "enthusiastic": "enthusiastic and passionate delivery",
    "knowledgeable": "deep subject knowledge",
    "expertise": "strong subject expertise",
    "research": "research-backed content",
    "responsive": "excellent student responsiveness",
    "available": "high availability for students",
    "helpful": "helpful and supportive approach",
    "office hours": "consistent office hours",
    "fair grading": "fair and transparent grading",
    "rubric": "clear assignment rubrics",
    "practical": "use of practical real-world examples",
    "real-world": "application of real-world context",
    "punctual": "punctuality and professionalism",
    "professional": "high professionalism",
    "motivating": "motivating and inspiring",
    "encourages": "encourages student participation",
}

# Negative aspect phrases mapped to human-readable improvement labels
_IMPROVEMENT_MAP: Dict[str, str] = {
    "unclear": "improve explanation clarity",
    "confusing": "improve explanation clarity",
    "fast": "reduce lecture pace",
    "too fast": "reduce lecture pace",
    "mumble": "improve speech clarity",
    "vague": "provide clearer instructions",
    "late": "improve punctuality and responsiveness",
    "delayed": "improve response time",
    "unresponsive": "improve response time",
    "boring": "increase classroom engagement",
    "monotonous": "vary teaching methods",
    "one-sided": "encourage more student interaction",
    "outdated": "update course content",
    "inaccurate": "verify factual accuracy",
    "errors": "verify factual accuracy",
    "excessive workload": "balance assignment workload",
    "too many assignments": "balance assignment workload",
    "grading": "improve grading consistency",
    "inconsistent": "improve grading consistency",
    "generic feedback": "provide more specific assignment feedback",
    "no feedback": "provide more specific assignment feedback",
    "no model answers": "share model answers post-submission",
    "disconnected": "align assignments with lecture content",
}


def extract_strengths(text: str) -> List[str]:
    """
    Extract strength phrases from a single feedback text.

    Uses keyword matching against curated positive phrase patterns.
    Returns de-duplicated list of human-readable strength labels.
    """
    lower = text.lower()
    found: List[str] = []
    seen_labels: set = set()
    for keyword, label in _STRENGTH_MAP.items():
        if keyword in lower and label not in seen_labels:
            found.append(label)
            seen_labels.add(label)
    return found[:5]


def extract_improvements(text: str) -> List[str]:
    """
    Extract improvement phrases from a single feedback text.

    Uses keyword matching against curated negative phrase patterns.
    Returns de-duplicated list of human-readable improvement labels.
    """
    lower = text.lower()
    found: List[str] = []
    seen_labels: set = set()
    for keyword, label in _IMPROVEMENT_MAP.items():
        if keyword in lower and label not in seen_labels:
            found.append(label)
            seen_labels.add(label)
    return found[:5]


def extract_strengths_batch(texts: List[str]) -> List[str]:
    """
    Extract and frequency-rank strengths across multiple feedback texts.
    Returns top-5 most commonly mentioned strength labels.
    """
    counter: Counter = Counter()
    for text in texts:
        for s in extract_strengths(text):
            counter[s] += 1
    return [s for s, _ in counter.most_common(5)]


def extract_improvements_batch(texts: List[str]) -> List[str]:
    """
    Extract and frequency-rank improvement areas across multiple feedback texts.
    Returns top-5 most commonly mentioned improvement labels.
    """
    counter: Counter = Counter()
    for text in texts:
        for i in extract_improvements(text):
            counter[i] += 1
    return [i for i, _ in counter.most_common(5)]


def extract_keywords(texts: List[str], top_n: int = 20) -> List[str]:
    """
    Frequency-ranked content keywords extracted from feedback texts.

    Parameters
    ----------
    texts  : list of raw feedback strings
    top_n  : number of top keywords to return

    Returns
    -------
    list[str]
    """
    STOP = {
        "the", "a", "an", "is", "in", "it", "of", "and", "to", "for",
        "that", "this", "was", "are", "be", "as", "at", "by", "we",
        "he", "she", "his", "her", "they", "them", "with", "from",
        "has", "have", "had", "not", "but", "or", "on", "so", "very",
        "my", "our", "your", "also", "more", "i", "me", "you", "can",
        "does", "do", "did", "will", "would", "could", "should", "been",
        "teacher", "professor", "class", "course", "student", "students",
        "always", "never", "every", "often", "sometimes", "which",
    }
    words: List[str] = []
    for text in texts:
        tokens = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        words.extend(w for w in tokens if w not in STOP)
    counter: Counter = Counter(words)
    return [w for w, _ in counter.most_common(top_n)]


# ── Executive report generator ────────────────────────────────────────────────
def generate_executive_summary(
    teacher_name: str,
    overall_score: float,
    grade: str,
    descriptor: str,
    sentiment_breakdown: Dict[str, float],
    aspect_scores: Dict[str, float],
    feedback_summary: str,
) -> str:
    """
    Produce a structured plain-text executive summary block for a teacher.

    Parameters
    ----------
    teacher_name       : display name
    overall_score      : 0–100 (dashboard scale) or 0–10 (pipeline scale)
    grade              : letter grade ("A+", "B", …)
    descriptor         : qualitative label ("Excellent", …)
    sentiment_breakdown: {Positive/Neutral/Negative → float %}
    aspect_scores      : {aspect_name → float score}
    feedback_summary   : pre-generated summary paragraph

    Returns
    -------
    str — formatted multi-line report block
    """
    top_aspects  = sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    weak_aspects = sorted(aspect_scores.items(), key=lambda x: x[1])[:2]

    top_str  = ", ".join(f"{a} ({s:.1f})" for a, s in top_aspects)  or "N/A"
    weak_str = ", ".join(f"{a} ({s:.1f})" for a, s in weak_aspects) or "N/A"

    pos = sentiment_breakdown.get("Positive", 0)
    neu = sentiment_breakdown.get("Neutral",  0)
    neg = sentiment_breakdown.get("Negative", 0)

    display_score = overall_score if overall_score > 10 else overall_score * 10
    return (
        f"EXECUTIVE SUMMARY — {teacher_name.upper()}\n"
        f"{'=' * 60}\n\n"
        f"OVERALL PERFORMANCE\n"
        f"  Score        : {display_score:.1f} / 100\n"
        f"  Grade        : {grade}  ({descriptor})\n\n"
        f"SENTIMENT OVERVIEW\n"
        f"  Positive     : {pos:.1f}%\n"
        f"  Neutral      : {neu:.1f}%\n"
        f"  Negative     : {neg:.1f}%\n\n"
        f"KEY STRENGTHS\n"
        f"  {top_str}\n\n"
        f"AREAS FOR IMPROVEMENT\n"
        f"  {weak_str}\n\n"
        f"FEEDBACK SUMMARY\n"
        f"  {feedback_summary}\n\n"
        f"{'=' * 60}"
    )


# ── Teacher Feedback Summarization Module Helpers & API ───────────────────────

_STRENGTH_TO_ASPECT: Dict[str, str] = {
    "clear explanations": "Communication",
    "articulate communication": "Communication",
    "well-structured lectures": "Communication",
    "highly engaging teaching style": "Engagement",
    "interactive classroom sessions": "Engagement",
    "enthusiastic and passionate delivery": "Engagement",
    "encourages student participation": "Engagement",
    "deep subject knowledge": "Subject Knowledge",
    "strong subject expertise": "Subject Knowledge",
    "research-backed content": "Subject Knowledge",
    "excellent student responsiveness": "Responsiveness",
    "high availability for students": "Responsiveness",
    "helpful and supportive approach": "Responsiveness",
    "consistent office hours": "Responsiveness",
    "fair and transparent grading": "Assignment Quality",
    "clear assignment rubrics": "Assignment Quality",
    "use of practical real-world examples": "Assignment Quality",
    "application of real-world context": "Assignment Quality",
    "punctuality and professionalism": "Responsiveness",
    "high professionalism": "Responsiveness",
    "motivating and inspiring": "Engagement",
}

_IMPROVEMENT_TO_ASPECT: Dict[str, str] = {
    "improve explanation clarity": "Communication",
    "reduce lecture pace": "Communication",
    "improve speech clarity": "Communication",
    "provide clearer instructions": "Communication",
    "improve punctuality and responsiveness": "Responsiveness",
    "improve response time": "Responsiveness",
    "increase classroom engagement": "Engagement",
    "vary teaching methods": "Engagement",
    "encourage more student interaction": "Engagement",
    "update course content": "Subject Knowledge",
    "verify factual accuracy": "Subject Knowledge",
    "balance assignment workload": "Assignment Quality",
    "improve grading consistency": "Assignment Quality",
    "provide more specific assignment feedback": "Assignment Quality",
    "share model answers post-submission": "Assignment Quality",
    "align assignments with lecture content": "Assignment Quality",
}


def _anonymize_student_mentions(text: str) -> str:
    """Filter out pronouns and student mentions to make text anonymous and generic."""
    # Student A / Student B / student X
    text = re.sub(r"\bStudent\s+[A-Z]\b", "Feedback", text, flags=re.IGNORECASE)
    # A student / one student / some students commented/says
    text = re.sub(r"\b(a|one|some)\s+students?\s+(says|commented|noted|notices|noticed|feels|feel|thinks|thought|writes|wrote)\b", "feedback indicates", text, flags=re.IGNORECASE)
    text = re.sub(r"\bstudents?\s+(says|commented|noted|notices|noticed|feels|feel|thinks|thought|writes|wrote)\b", "feedback indicates", text, flags=re.IGNORECASE)
    # "my students" -> "the class" or "students"
    text = re.sub(r"\bmy\s+students\b", "the class", text, flags=re.IGNORECASE)
    # First person pronoun cleanup
    text = re.sub(r"\bI\s+(feel|think|believe|found)\b", "it was noted", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI\s+would\s+like\b", "it was suggested", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI\s+(loved|liked|enjoyed|appreciated)\b", "feedback appreciated", text, flags=re.IGNORECASE)
    text = re.sub(r"\bin\s+my\s+opinion\b", "feedback indicates", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfor\s+me\b", "", text, flags=re.IGNORECASE)
    # Standalone pronouns
    text = re.sub(r"\bI\b", "feedback", text)
    text = re.sub(r"\bmy\b", "the", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwe\b", "feedback", text, flags=re.IGNORECASE)
    return text.strip()


def _summarize_low_volume(comments: List[str], aspect_scores: Dict[str, float]) -> str:
    """Lightweight summary for small datasets to prevent overgeneralization."""
    n = len(comments)
    if n == 0:
        return "No feedback comments available."
    cleaned = [_anonymize_student_mentions(c) for c in comments if c.strip()]
    combined = " ".join(cleaned)
    words = combined.split()
    if len(words) > 60:
        combined = " ".join(words[:60]) + "..."
    prefix = f"Based on a limited sample of {n} reviews, feedback notes that: {combined}"
    pref_words = prefix.split()
    if len(pref_words) > 78:
        prefix = " ".join(pref_words[:77]) + "..."
    return prefix


def _generate_extractive_summary(
    anon_comments: List[str],
    aspect_scores: Dict[str, float],
) -> str:
    """Generate summary by selecting sentences that match strong and weak aspects."""
    # Scale to 0-10
    norm_scores = {}
    for aspect, score in aspect_scores.items():
        norm_scores[aspect] = score / 10.0 if score > 10.0 else score

    sorted_aspects = sorted(norm_scores.items(), key=lambda x: x[1])
    weak_aspects = [asp for asp, score in sorted_aspects if score <= 6.5][:2]
    strong_aspects = [asp for asp, score in reversed(sorted_aspects) if score >= 7.0][:2]

    sentences = []
    for c in anon_comments:
        for s in re.split(r"(?<=[.!?])\s+", c.strip()):
            s_clean = s.strip()
            if len(s_clean) > 15:
                sentences.append(s_clean)

    aspect_keywords = {
        "Communication": ["explain", "clear", "clarity", "lecture", "speak", "pace", "fast", "understand"],
        "Subject Knowledge": ["knowledge", "expert", "research", "accurate", "mistake", "error", "subject"],
        "Engagement": ["engage", "interactive", "fun", "enthusias", "participat", "interest", "boring"],
        "Responsiveness": ["responsive", "reply", "email", "office hours", "available", "help", "support"],
        "Assignment Quality": ["assignment", "grading", "grade", "rubric", "workload", "deadline", "exam"],
    }

    strong_sentences = []
    weak_sentences = []
    neutral_sentences = []

    for s in sentences:
        s_lower = s.lower()
        has_strong = any(any(kw in s_lower for kw in aspect_keywords.get(asp, [])) for asp in strong_aspects)
        has_weak = any(any(kw in s_lower for kw in aspect_keywords.get(asp, [])) for asp in weak_aspects)
        if has_strong and not has_weak:
            strong_sentences.append(s)
        elif has_weak and not has_strong:
            weak_sentences.append(s)
        else:
            neutral_sentences.append(s)

    selected = []
    if strong_sentences:
        selected.append(strong_sentences[0])
    if weak_sentences:
        selected.append(weak_sentences[0])
    if len(selected) < 3 and neutral_sentences:
        selected.append(neutral_sentences[0])
    if len(selected) < 3 and len(strong_sentences) > 1:
        selected.append(strong_sentences[1])
    if len(selected) < 3 and len(weak_sentences) > 1:
        selected.append(weak_sentences[1])

    summary = " ".join(selected)
    words = summary.split()
    if len(words) > 75:
        summary = " ".join(words[:75]) + "..."
    return summary


def summarize_teacher_feedback(
    comments: List[str],
    aspect_scores: Dict[str, float],
    model_name: str = PRIMARY_MODEL,
) -> Dict[str, Any]:
    """
    Generate a concise feedback summary, strengths, and areas for improvement.

    Guarantees:
      - Summary is under 80 words.
      - Mentions of individual students are anonymized.
      - Low review counts (< 5) are flagged to prevent overgeneralization.
      - Aligned with aggregated aspect scores.
    """
    if not comments:
        return {
            "summary": "No feedback comments available.",
            "strengths": [],
            "improvements": [],
        }

    # Normalize aspect scores to 0-10
    norm_scores = {}
    for aspect, score in aspect_scores.items():
        norm_scores[aspect] = score / 10.0 if score > 10.0 else score

    # 1. Strengths & Improvements extraction (aligned with aspect scores)
    raw_strengths = extract_strengths_batch(comments)
    raw_improvements = extract_improvements_batch(comments)

    # Sort strengths by aspect score (descending)
    def strength_sort_key(s):
        asp = _STRENGTH_TO_ASPECT.get(s, "")
        return norm_scores.get(asp, 5.0)

    # Sort improvements by aspect score (ascending)
    def improvement_sort_key(i):
        asp = _IMPROVEMENT_TO_ASPECT.get(i, "")
        return norm_scores.get(asp, 5.0)

    strengths = sorted(raw_strengths, key=strength_sort_key, reverse=True)[:3]
    improvements = sorted(raw_improvements, key=improvement_sort_key)[:3]

    # 2. Summary Generation
    if len(comments) < 5:
        summary = _summarize_low_volume(comments, aspect_scores)
    else:
        anon_comments = [_anonymize_student_mentions(c) for c in comments if c.strip()]
        combined_text = " ".join(anon_comments)[:3500]

        summary = ""
        pipe = _load_summarizer(model_name)
        if pipe is not None:
            try:
                # Set strict token limits to generate < 80 words
                result = pipe(
                    combined_text,
                    min_length=15,
                    max_length=50,
                    do_sample=False,
                )
                raw_summary = result[0]["summary_text"].strip()
                summary = _anonymize_student_mentions(raw_summary)
            except Exception as exc:
                logger.warning("Abstractive summarizer failed: %s. Using extractive.", exc)

        # Fallback to extractive if abstractive failed or returned empty
        if not summary:
            summary = _generate_extractive_summary(anon_comments, aspect_scores)

    # Final post-processing word count guard
    words = summary.split()
    if len(words) > 78:
        summary = " ".join(words[:77]) + "..."

    return {
        "summary": summary,
        "strengths": strengths,
        "improvements": improvements,
    }
