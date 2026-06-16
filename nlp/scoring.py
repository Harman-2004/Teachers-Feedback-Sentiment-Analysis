"""
nlp/scoring.py
--------------
Composite performance scoring engine — produces the canonical pipeline JSON:

{
    "overall_score":           7.8,    # 0–10
    "communication_score":     9.1,
    "subject_knowledge_score": 8.5,
    "engagement_score":        7.4,
    "responsiveness_score":    7.0,
    "assignment_quality_score":4.8,
    "strengths":               ["clear explanations"],
    "improvements":            ["assignment workload"],
    "confidence":              0.89
}

Public functions
----------------
score_single(text, sent_result, aspect_result)  →  pipeline dict (single text)
score_batch(texts, sent_results, aspect_results) →  list[pipeline dict]
compute_teacher_score(...)                       →  dashboard dict (0–100 scale)
rank_teachers(...)                               →  ranked list
trend_analysis(...)                              →  DataFrame
score_distribution_bins(...)                     →  (edges, counts)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Pipeline aspect keys ─────────────────────────────────────────────────────
PIPELINE_ASPECTS = [
    "Communication",
    "Subject Knowledge",
    "Engagement",
    "Responsiveness",
    "Assignment Quality",
]

# Pipeline JSON key names (snake_case)
_ASPECT_KEY: Dict[str, str] = {
    "Communication":      "communication_score",
    "Subject Knowledge":  "subject_knowledge_score",
    "Engagement":         "engagement_score",
    "Responsiveness":     "responsiveness_score",
    "Assignment Quality": "assignment_quality_score",
}

# Weights for overall_score (must sum to 1.0)
ASPECT_WEIGHTS: Dict[str, float] = {
    "Communication":      0.22,
    "Subject Knowledge":  0.25,
    "Engagement":         0.20,
    "Responsiveness":     0.18,
    "Assignment Quality": 0.15,
}

# ── Grade thresholds (dashboard 0–100 scale) ─────────────────────────────────
GRADE_THRESHOLDS = [
    (90, "A+", "Outstanding"),
    (80, "A",  "Excellent"),
    (70, "B",  "Good"),
    (60, "C",  "Satisfactory"),
    (50, "D",  "Needs Improvement"),
    (0,  "F",  "Poor"),
]

# ── Strength / improvement keyword clusters ───────────────────────────────────
_STRENGTH_SIGNALS: Dict[str, List[str]] = {
    "clear explanations": [
        "clear", "clarity", "explains", "articulate", "concise", "simple",
        "well-structured", "understandable", "easy to follow",
    ],
    "deep subject knowledge": [
        "knowledge", "expertise", "mastery", "research", "scholar",
        "informed", "domain", "authority", "accurate",
    ],
    "high student engagement": [
        "engaging", "interactive", "participation", "enthusiastic",
        "dynamic", "fun", "energetic", "motivating", "interesting",
    ],
    "strong responsiveness": [
        "responsive", "available", "reply", "quick", "office hours",
        "support", "helpful", "feedback", "accessible", "attentive",
    ],
    "excellent assignment design": [
        "assignments", "rubric", "fair", "challenging", "relevant",
        "practical", "project", "well-designed", "structured",
    ],
    "effective communication": [
        "communicates", "speaks", "language", "tone", "articulation",
        "narrates", "explains", "pace", "diction",
    ],
    "professional and punctual": [
        "punctual", "professional", "respectful", "dedicated",
        "committed", "on time", "responsible",
    ],
}

_IMPROVEMENT_SIGNALS: Dict[str, List[str]] = {
    "assignment workload": [
        "excessive", "workload", "overloaded", "too much", "overwhelming",
        "tight deadline", "unrealistic", "heavy",
    ],
    "lecture clarity": [
        "unclear", "confusing", "hard to follow", "jumps", "disjointed",
        "vague", "mumbles", "fast",
    ],
    "student interaction": [
        "no interaction", "one-sided", "boring", "passive", "monotonous",
        "no participation", "disengaged", "dull",
    ],
    "feedback quality": [
        "generic feedback", "no feedback", "late marks", "vague feedback",
        "one-word", "no model answers", "inconsistent grading",
    ],
    "responsiveness": [
        "delayed", "slow reply", "unavailable", "ignores",
        "dismisses", "not accessible", "no office hours",
    ],
    "subject depth": [
        "outdated", "inaccurate", "errors", "wrong", "limited knowledge",
        "reads notes", "textbook only", "shallow",
    ],
    "classroom energy": [
        "lifeless", "unenthusiastic", "demoralizing", "tense",
        "unwelcoming", "no energy", "disinterested",
    ],
}


def _extract_signals(
    text: str, signal_map: Dict[str, List[str]]
) -> List[str]:
    """
    Return labels whose keywords appear in *text*.
    Normalises text to lowercase before matching.
    """
    lower = text.lower()
    found: List[str] = []
    for label, keywords in signal_map.items():
        if any(kw in lower for kw in keywords):
            found.append(label)
    return found


def _blend_scores(
    polarity: float,           # 0–10 from sentiment.py
    aspect_score: float,       # 0–10 from aspects.py
    aspect_relevance: float,   # 0–1  how strongly aspect is present
    alpha: float = 0.40,       # weight of polarity
) -> float:
    """
    Blend sentiment polarity with aspect-specific ABSA score.

    When the aspect is not detected (low relevance), fall back more
    heavily on the overall sentiment polarity.
    """
    # Linearly interpolate: high relevance → trust ABSA more
    beta = min(1.0, aspect_relevance * 2.0)   # 0 → 1 when rel ≥ 0.5
    blended = (1.0 - beta) * polarity + beta * aspect_score
    # Mix with overall polarity at alpha level for regularisation
    final = (1.0 - alpha) * blended + alpha * polarity
    return round(max(0.0, min(10.0, final)), 4)


# ── Primary pipeline output ───────────────────────────────────────────────────
def score_single(
    text: str,
    sentiment_result: Dict[str, Any],
    aspect_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce the canonical pipeline JSON for **one** feedback text.

    Parameters
    ----------
    text             : original feedback string
    sentiment_result : output dict from ``sentiment.analyze_sentiment`` for this text
    aspect_result    : output dict from ``aspects.score_aspects`` for this text

    Returns
    -------
    dict (canonical JSON format)
        overall_score            float  0–10
        communication_score      float  0–10
        subject_knowledge_score  float  0–10
        engagement_score         float  0–10
        responsiveness_score     float  0–10
        assignment_quality_score float  0–10
        strengths                list[str]
        improvements             list[str]
        confidence               float  0–1
        sentiment_label          str
        is_mixed                 bool
    """
    polarity   = sentiment_result.get("polarity", 5.0)
    confidence = sentiment_result.get("confidence", 0.5)
    is_mixed   = sentiment_result.get("is_mixed", False)
    label      = sentiment_result.get("label", "Neutral")

    aspect_scores_10: Dict[str, float] = {}
    for asp in PIPELINE_ASPECTS:
        asp_info    = aspect_result.get(asp, {})
        asp_score   = asp_info.get("score_10", polarity)   # fallback to polarity
        asp_rel     = asp_info.get("relevance", 0.0)
        blended     = _blend_scores(polarity, asp_score, asp_rel)
        aspect_scores_10[asp] = blended

    # Weighted overall
    overall = sum(
        aspect_scores_10[a] * ASPECT_WEIGHTS[a]
        for a in PIPELINE_ASPECTS
    )
    overall = round(max(0.0, min(10.0, overall)), 2)

    # Strengths & improvements
    strengths    = _extract_signals(text, _STRENGTH_SIGNALS)
    improvements = _extract_signals(text, _IMPROVEMENT_SIGNALS)

    # If no keyword signals found, infer from aspect scores
    if not strengths:
        best_asp = max(PIPELINE_ASPECTS, key=lambda a: aspect_scores_10[a])
        if aspect_scores_10[best_asp] >= 6.5:
            strengths = [best_asp.lower()]
    if not improvements:
        worst_asp = min(PIPELINE_ASPECTS, key=lambda a: aspect_scores_10[a])
        if aspect_scores_10[worst_asp] < 5.0:
            improvements = [worst_asp.lower()]

    return {
        "overall_score":            overall,
        "communication_score":      round(aspect_scores_10["Communication"],      2),
        "subject_knowledge_score":  round(aspect_scores_10["Subject Knowledge"],  2),
        "engagement_score":         round(aspect_scores_10["Engagement"],          2),
        "responsiveness_score":     round(aspect_scores_10["Responsiveness"],      2),
        "assignment_quality_score": round(aspect_scores_10["Assignment Quality"], 2),
        "strengths":    strengths[:5],
        "improvements": improvements[:5],
        "confidence":   round(confidence, 4),
        "sentiment_label": label,
        "is_mixed":        is_mixed,
    }


def score_batch(
    texts: List[str],
    sentiment_results: List[Dict[str, Any]],
    aspect_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Produce canonical pipeline JSON for a **batch** of texts.

    Parameters
    ----------
    texts             : list of raw feedback strings
    sentiment_results : list[dict] from ``sentiment.analyze_sentiment``
    aspect_results    : list[dict] from ``aspects.score_aspects_batch``
                        Each dict must have an ``"aspects"`` key.

    Returns
    -------
    list[dict]  — one canonical dict per input text
    """
    output: List[Dict[str, Any]] = []
    for text, sent_r, asp_r_wrap in zip(texts, sentiment_results, aspect_results):
        asp_r = asp_r_wrap.get("aspects", asp_r_wrap)
        output.append(score_single(text, sent_r, asp_r))
    return output


def aggregate_batch_scores(
    batch_output: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate a batch of score_single outputs into a teacher-level summary.

    Returns
    -------
    dict
        Same keys as score_single plus:
        n_feedback      int
        grade           str
        descriptor      str
        sentiment_breakdown  dict
    """
    if not batch_output:
        return {}

    fields = [
        "overall_score", "communication_score", "subject_knowledge_score",
        "engagement_score", "responsiveness_score", "assignment_quality_score",
        "confidence",
    ]
    aggregated: Dict[str, Any] = {}
    for f in fields:
        vals = [r[f] for r in batch_output if f in r]
        aggregated[f] = round(float(np.mean(vals)), 2) if vals else 0.0

    # Combine strengths / improvements (frequency-ranked)
    from collections import Counter
    str_counter = Counter(s for r in batch_output for s in r.get("strengths", []))
    imp_counter = Counter(i for r in batch_output for i in r.get("improvements", []))
    aggregated["strengths"]    = [s for s, _ in str_counter.most_common(5)]
    aggregated["improvements"] = [i for i, _ in imp_counter.most_common(5)]

    # Grade on 0–100 scale
    score_100 = aggregated["overall_score"] * 10
    grade, descriptor = "F", "Poor"
    for threshold, g, d in GRADE_THRESHOLDS:
        if score_100 >= threshold:
            grade, descriptor = g, d
            break
    aggregated["grade"]       = grade
    aggregated["descriptor"]  = descriptor
    aggregated["n_feedback"]  = len(batch_output)

    # Sentiment breakdown
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in batch_output:
        lbl = r.get("sentiment_label", "Neutral")
        counts[lbl] = counts.get(lbl, 0) + 1
    n = len(batch_output)
    aggregated["sentiment_breakdown"] = {k: round(v / n * 100, 2) for k, v in counts.items()}

    return aggregated


# ── Dashboard backward-compat functions ──────────────────────────────────────
def compute_teacher_score(
    sentiment_results: List[Dict[str, Any]],
    aspect_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Dashboard-compatible aggregate scorer (0–100 scale output).

    Parameters
    ----------
    sentiment_results : output of ``sentiment.analyze_sentiment``
    aspect_results    : output of ``aspects.analyze_aspects_batch`` (optional)
    """
    if not sentiment_results:
        return {
            "overall_score": 0.0, "grade": "N/A",
            "descriptor": "No data", "aspect_scores": {},
            "sentiment_breakdown": {},
        }

    # Build aspect_scores dict (0–100) if aspect_results provided
    aspect_scores_100: Dict[str, float] = {}
    if aspect_results:
        # Use new score_aspects output if available, else old detect_aspects
        for sent_r, asp_wrap in zip(sentiment_results, aspect_results):
            asp_dict = asp_wrap.get("aspects", {})
            polarity = sent_r.get("polarity", 5.0)
            for asp in PIPELINE_ASPECTS:
                asp_info = asp_dict.get(asp, {})
                if isinstance(asp_info, dict) and "score_10" in asp_info:
                    s10 = _blend_scores(
                        polarity,
                        asp_info.get("score_10", polarity),
                        asp_info.get("relevance", 0.0),
                    )
                else:
                    s10 = polarity
                aspect_scores_100.setdefault(asp, []).append(s10 * 10)  # type: ignore[arg-type]

        aspect_scores_100 = {
            k: round(float(np.mean(v)), 2)
            for k, v in aspect_scores_100.items()
        }

    # Overall score (0–100)
    if aspect_scores_100:
        overall = sum(
            aspect_scores_100.get(a, 50.0) * ASPECT_WEIGHTS[a]
            for a in PIPELINE_ASPECTS
        ) * 10  # already in 0–100
    else:
        polarities = [r.get("polarity", 5.0) for r in sentiment_results]
        overall = float(np.mean(polarities)) * 10

    overall = round(min(100.0, max(0.0, overall)), 2)

    grade, descriptor = "F", "Poor"
    for threshold, g, d in GRADE_THRESHOLDS:
        if overall >= threshold:
            grade, descriptor = g, d
            break

    # Sentiment breakdown
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in sentiment_results:
        lbl = r.get("label", "Neutral")
        counts[lbl] = counts.get(lbl, 0) + 1
    n = len(sentiment_results)
    breakdown = {k: round(v / n * 100, 2) for k, v in counts.items()}

    return {
        "overall_score":      overall,
        "grade":              grade,
        "descriptor":         descriptor,
        "aspect_scores":      aspect_scores_100,
        "sentiment_breakdown": breakdown,
    }


def rank_teachers(teacher_scores: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = [
        {
            "teacher":    name,
            "score":      data.get("overall_score", 0.0),
            "grade":      data.get("grade", "N/A"),
            "descriptor": data.get("descriptor", ""),
            "sentiment_breakdown": data.get("sentiment_breakdown", {}),
        }
        for name, data in teacher_scores.items()
    ]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(ranked):
        item["rank"] = i + 1
    return ranked


def trend_analysis(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
    freq: str = "ME",
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, score_col])
    df = df.set_index(date_col)
    resampled = df[score_col].resample(freq).agg(["mean", "count"]).reset_index()
    resampled.columns = ["date", "mean_score", "count"]
    resampled["mean_score"] = resampled["mean_score"].round(2)
    return resampled


def score_distribution_bins(
    scores: List[float], n_bins: int = 10
) -> Tuple[List[float], List[int]]:
    arr = np.array(scores)
    counts_arr, bin_edges = np.histogram(arr, bins=n_bins, range=(0, 100))
    return bin_edges.tolist(), counts_arr.tolist()
