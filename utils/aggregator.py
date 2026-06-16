"""
utils/aggregator.py
-------------------
Score Aggregation Module — Teacher Feedback Analytics Pipeline.

Responsibility
--------------
Takes a DataFrame of raw or pre-scored feedback records, runs the NLP
pipeline where needed, and returns **one aggregated row per teacher**
containing weighted composite scores, statistical metadata, and
confidence metrics.

Weights (configurable via AggregationConfig)
--------------------------------------------
  Communication       25 %
  Subject Knowledge   25 %
  Engagement          20 %
  Responsiveness      15 %
  Assignment Quality  15 %
  ─────────────────────────
  Total              100 %

Output DataFrame columns (one row per teacher)
----------------------------------------------
  teacher_name              str
  subject                   str   (most common subject for teacher)
  total_reviews             int
  communication_score       float   0 – 10
  subject_knowledge_score   float   0 – 10
  engagement_score          float   0 – 10
  responsiveness_score      float   0 – 10
  assignment_quality_score  float   0 – 10
  overall_score             float   0 – 10  (weighted composite)
  confidence_score          float   0 –  1  (mean model confidence)
  confidence_band           str   "High" | "Medium" | "Low"
  grade                     str   "A+" … "F"
  descriptor                str   "Outstanding" … "Poor"
  positive_pct              float   % of positive reviews
  neutral_pct               float
  negative_pct              float
  mixed_count               int     reviews flagged as mixed-sentiment
  strengths                 str   top strengths joined by " | "
  improvements              str   top improvements joined by " | "
  rank                      int   1 = best teacher

Public API
----------
  aggregate(df, config)               → aggregated DataFrame
  aggregate_from_scores(scored_df)    → aggregated DataFrame (pre-scored input)
  run_pipeline_on_df(df, config)      → df with all NLP score columns added
  pivot_aspect_scores(agg_df)         → long-format DataFrame for plotting
  teacher_report_dict(agg_df, name)   → dict for a single teacher (report export)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Score column names produced by the NLP pipeline ──────────────────────────
SCORE_COLS: List[str] = [
    "communication_score",
    "subject_knowledge_score",
    "engagement_score",
    "responsiveness_score",
    "assignment_quality_score",
]

# Map aspect label → DataFrame column
ASPECT_COL: Dict[str, str] = {
    "Communication":      "communication_score",
    "Subject Knowledge":  "subject_knowledge_score",
    "Engagement":         "engagement_score",
    "Responsiveness":     "responsiveness_score",
    "Assignment Quality": "assignment_quality_score",
}

# Grade thresholds on a 0–10 scale
_GRADE_THRESHOLDS = [
    (9.0, "A+", "Outstanding"),
    (8.0, "A",  "Excellent"),
    (7.0, "B",  "Good"),
    (6.0, "C",  "Satisfactory"),
    (5.0, "D",  "Needs Improvement"),
    (0.0, "F",  "Poor"),
]


# ── Configuration ──────────────────────────────────────────────────────────────
@dataclass
class AggregationConfig:
    """
    Controls weights and behaviour of the score aggregator.

    Attributes
    ----------
    weights : dict
        Aspect label → weight (must sum to 1.0).
    pipeline_batch_size : int
        Batch size passed to the NLP pipeline when scoring raw text.
    min_reviews_for_high_confidence : int
        A teacher needs at least this many reviews for "High" confidence.
    min_reviews_for_medium_confidence : int
        Below this threshold the band is "Low".
    """
    weights: Dict[str, float] = field(default_factory=lambda: {
        "Communication":      0.25,
        "Subject Knowledge":  0.25,
        "Engagement":         0.20,
        "Responsiveness":     0.15,
        "Assignment Quality": 0.15,
    })
    pipeline_batch_size: int = 16
    min_reviews_for_high_confidence: int = 10
    min_reviews_for_medium_confidence: int = 3

    def validate(self) -> None:
        total = sum(self.weights.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"Weights must sum to 1.0 (current sum = {total:.4f}). "
                f"Check AggregationConfig.weights."
            )


_DEFAULT_CONFIG = AggregationConfig()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _grade(score_10: float) -> tuple[str, str]:
    """Return (letter_grade, descriptor) for a 0–10 score."""
    for threshold, grade, descriptor in _GRADE_THRESHOLDS:
        if score_10 >= threshold:
            return grade, descriptor
    return "F", "Poor"


def _confidence_band(
    mean_confidence: float,
    total_reviews: int,
    cfg: AggregationConfig,
) -> str:
    """
    Classify the aggregate confidence as High / Medium / Low.

    Two dimensions:
      - Model confidence (mean of per-review confidence scores)
      - Review volume (more reviews = more reliable estimate)
    """
    volume_ok_high   = total_reviews >= cfg.min_reviews_for_high_confidence
    volume_ok_medium = total_reviews >= cfg.min_reviews_for_medium_confidence

    if mean_confidence >= 0.75 and volume_ok_high:
        return "High"
    elif mean_confidence >= 0.55 and volume_ok_medium:
        return "Medium"
    return "Low"


def _normalise_to_10(series: pd.Series) -> pd.Series:
    """
    Min-max normalise a series to [0, 10].

    If all values are identical (zero variance), returns the series unchanged
    (already in [0, 10] from the pipeline).
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series.clip(0, 10)
    return ((series - lo) / (hi - lo) * 10).clip(0, 10)


def _top_n_from_lists(series: pd.Series, n: int = 5) -> str:
    """
    Frequency-rank items across a Series of lists (or pipe-separated strings)
    and return the top-n joined by ' | '.
    """
    counter: Counter = Counter()
    for cell in series.dropna():
        if isinstance(cell, list):
            items = cell
        elif isinstance(cell, str):
            items = [x.strip() for x in cell.split("|") if x.strip()]
        else:
            continue
        counter.update(items)
    top = [item for item, _ in counter.most_common(n)]
    return " | ".join(top) if top else ""


def _most_common(series: pd.Series) -> Any:
    """Return the most frequent non-null value in a series."""
    counts = series.dropna().value_counts()
    return counts.index[0] if len(counts) > 0 else None


# ── Step 1: Run pipeline on raw DataFrame ─────────────────────────────────────
def run_pipeline_on_df(
    df: pd.DataFrame,
    config: Optional[AggregationConfig] = None,
) -> pd.DataFrame:
    """
    Score every row in *df* using the NLP pipeline and attach all score
    columns to the DataFrame.

    Parameters
    ----------
    df     : DataFrame with at least a ``feedback_text`` column.
    config : AggregationConfig (uses defaults if None).

    Returns
    -------
    pd.DataFrame — original columns PLUS:
        overall_score, communication_score, subject_knowledge_score,
        engagement_score, responsiveness_score, assignment_quality_score,
        confidence, sentiment_label, is_mixed, strengths_list, improvements_list
    """
    cfg = config or _DEFAULT_CONFIG
    cfg.validate()

    if "feedback_text" not in df.columns:
        raise ValueError("DataFrame must contain a 'feedback_text' column.")

    texts = df["feedback_text"].astype(str).tolist()

    logger.info("Running NLP pipeline on %d rows...", len(texts))
    from nlp.pipeline import analyze_batch, PipelineConfig

    pipe_cfg = PipelineConfig(batch_size=cfg.pipeline_batch_size)
    results = analyze_batch(texts, config=pipe_cfg)

    # Flatten results into columns
    rows_extra = []
    for r in results:
        rows_extra.append({
            "overall_score":            r.get("overall_score",            5.0),
            "communication_score":      r.get("communication_score",      5.0),
            "subject_knowledge_score":  r.get("subject_knowledge_score",  5.0),
            "engagement_score":         r.get("engagement_score",         5.0),
            "responsiveness_score":     r.get("responsiveness_score",     5.0),
            "assignment_quality_score": r.get("assignment_quality_score", 5.0),
            "confidence":               r.get("confidence",               0.5),
            "sentiment_label":          r.get("sentiment_label",          "Neutral"),
            "is_mixed":                 r.get("is_mixed",                 False),
            "strengths_list":           r.get("strengths",                []),
            "improvements_list":        r.get("improvements",             []),
        })

    extra_df = pd.DataFrame(rows_extra, index=df.index)

    # Drop any existing score columns to avoid duplication
    drop_cols = [c for c in extra_df.columns if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    return pd.concat([df.reset_index(drop=True), extra_df.reset_index(drop=True)], axis=1)


# ── Step 2: Aggregate pre-scored DataFrame by teacher ─────────────────────────
def aggregate_from_scores(
    scored_df: pd.DataFrame,
    config: Optional[AggregationConfig] = None,
) -> pd.DataFrame:
    """
    Aggregate a *pre-scored* DataFrame (must contain all SCORE_COLS) into
    one row per teacher.

    Parameters
    ----------
    scored_df : DataFrame with score columns already present (e.g. output
                of ``run_pipeline_on_df``).  Must contain:
                  - All columns in SCORE_COLS (0–10 floats)
                  - ``teacher_name`` (groups rows into teachers)
                  - ``confidence``   (model confidence per row)
                Optional but used when present:
                  - ``sentiment_label``, ``is_mixed``
                  - ``subject``, ``date``, ``semester``, ``rating``
                  - ``strengths_list``, ``improvements_list``

    config    : AggregationConfig (uses defaults if None).

    Returns
    -------
    pd.DataFrame  — one row per teacher, sorted by overall_score descending.
    """
    cfg = config or _DEFAULT_CONFIG
    cfg.validate()

    df = scored_df.copy()

    # Ensure teacher_name column exists
    if "teacher_name" not in df.columns:
        df["teacher_name"] = "Unknown Teacher"
        logger.warning("No 'teacher_name' column found; treating all rows as one teacher.")

    # Fill missing score columns with 5.0 (neutral mid-point)
    for col in SCORE_COLS:
        if col not in df.columns:
            df[col] = 5.0
            logger.warning("Missing score column '%s' — filled with 5.0.", col)
    if "confidence" not in df.columns:
        df["confidence"] = 0.5

    # ── Group by teacher ──────────────────────────────────────────────────────
    records: List[Dict[str, Any]] = []

    for teacher, group in df.groupby("teacher_name", sort=False):
        n = len(group)

        # ── Per-aspect means ──────────────────────────────────────────────────
        raw_scores: Dict[str, float] = {
            asp: float(group[col].mean())
            for asp, col in ASPECT_COL.items()
            if col in group.columns
        }

        # ── Normalise each aspect mean to [0, 10] ────────────────────────────
        # We normalise across ALL teachers later (cross-teacher normalisation)
        # so we just store the raw means here and do cross-teacher normalisation
        # after collecting all teachers.
        norm_scores: Dict[str, float] = {
            asp: round(max(0.0, min(10.0, score)), 4)
            for asp, score in raw_scores.items()
        }

        # ── Weighted overall score ────────────────────────────────────────────
        overall = sum(
            norm_scores.get(asp, 5.0) * cfg.weights.get(asp, 0.0)
            for asp in cfg.weights
        )
        overall = round(max(0.0, min(10.0, overall)), 4)

        # ── Confidence ────────────────────────────────────────────────────────
        mean_conf = float(group["confidence"].mean())

        # Penalise with Wilson-style shrinkage: more reviews → trust more
        # confidence_score = model_conf × volume_factor
        volume_factor = min(1.0, n / cfg.min_reviews_for_high_confidence)
        confidence_score = round(mean_conf * (0.6 + 0.4 * volume_factor), 4)

        band = _confidence_band(mean_conf, n, cfg)

        # ── Sentiment breakdown ───────────────────────────────────────────────
        pos_pct = neu_pct = neg_pct = 0.0
        mixed_count = 0
        if "sentiment_label" in group.columns:
            label_counts = group["sentiment_label"].value_counts()
            pos_pct = round(label_counts.get("Positive", 0) / n * 100, 2)
            neu_pct = round(label_counts.get("Neutral",  0) / n * 100, 2)
            neg_pct = round(label_counts.get("Negative", 0) / n * 100, 2)
        if "is_mixed" in group.columns:
            mixed_count = int(group["is_mixed"].sum())

        # ── Grade ─────────────────────────────────────────────────────────────
        grade, descriptor = _grade(overall)

        # ── Strengths & improvements ──────────────────────────────────────────
        strengths_str   = ""
        improvements_str = ""
        if "strengths_list" in group.columns:
            strengths_str = _top_n_from_lists(group["strengths_list"])
        if "improvements_list" in group.columns:
            improvements_str = _top_n_from_lists(group["improvements_list"])

        # ── Optional metadata ─────────────────────────────────────────────────
        subject  = _most_common(group["subject"])  if "subject"  in group.columns else None
        semester = _most_common(group["semester"]) if "semester" in group.columns else None

        row: Dict[str, Any] = {
            "teacher_name":             teacher,
            "subject":                  subject,
            "semester":                 semester,
            "total_reviews":            n,
            # ── Per-aspect scores (0–10) ─────────────────────────────────────
            "communication_score":      round(norm_scores.get("Communication",      5.0), 2),
            "subject_knowledge_score":  round(norm_scores.get("Subject Knowledge",  5.0), 2),
            "engagement_score":         round(norm_scores.get("Engagement",         5.0), 2),
            "responsiveness_score":     round(norm_scores.get("Responsiveness",     5.0), 2),
            "assignment_quality_score": round(norm_scores.get("Assignment Quality", 5.0), 2),
            # ── Composite ────────────────────────────────────────────────────
            "overall_score":            round(overall, 2),
            # ── Confidence ───────────────────────────────────────────────────
            "confidence_score":         confidence_score,
            "confidence_band":          band,
            # ── Grade ────────────────────────────────────────────────────────
            "grade":                    grade,
            "descriptor":               descriptor,
            # ── Sentiment breakdown ───────────────────────────────────────────
            "positive_pct":             pos_pct,
            "neutral_pct":              neu_pct,
            "negative_pct":             neg_pct,
            "mixed_count":              mixed_count,
            # ── Qualitative ──────────────────────────────────────────────────
            "strengths":                strengths_str,
            "improvements":             improvements_str,
        }
        records.append(row)

    if not records:
        return pd.DataFrame()

    agg_df = pd.DataFrame(records)

    # ── Cross-teacher normalisation ───────────────────────────────────────────
    # Rescale aspect scores so the best teacher in each dimension gets 10.0
    # and the worst gets the proportionally lower score (preserves rank order).
    for col in SCORE_COLS:
        if col in agg_df.columns and agg_df[col].nunique() > 1:
            agg_df[col] = _normalise_to_10(agg_df[col]).round(2)

    # Recompute overall after normalisation
    agg_df["overall_score"] = (
        agg_df["communication_score"]      * cfg.weights["Communication"]
        + agg_df["subject_knowledge_score"]  * cfg.weights["Subject Knowledge"]
        + agg_df["engagement_score"]         * cfg.weights["Engagement"]
        + agg_df["responsiveness_score"]     * cfg.weights["Responsiveness"]
        + agg_df["assignment_quality_score"] * cfg.weights["Assignment Quality"]
    ).clip(0, 10).round(2)

    # Recompute grade after re-normalisation
    agg_df[["grade", "descriptor"]] = agg_df["overall_score"].apply(
        lambda s: pd.Series(_grade(s))
    )

    # ── Rank ──────────────────────────────────────────────────────────────────
    agg_df = agg_df.sort_values("overall_score", ascending=False).reset_index(drop=True)
    agg_df["rank"] = agg_df.index + 1

    logger.info(
        "Aggregation complete: %d teachers from %d feedback records.",
        len(agg_df),
        len(df),
    )
    return agg_df


# ── Step 1 + 2 combined ────────────────────────────────────────────────────────
def aggregate(
    df: pd.DataFrame,
    config: Optional[AggregationConfig] = None,
) -> pd.DataFrame:
    """
    **Primary entry point.**

    Accepts a raw feedback DataFrame (with ``feedback_text`` and optionally
    ``teacher_name``), runs the NLP pipeline on every row, then aggregates
    into one row per teacher.

    Parameters
    ----------
    df     : raw feedback DataFrame (at minimum: ``feedback_text`` column)
    config : AggregationConfig (uses defaults if None)

    Returns
    -------
    pd.DataFrame  — one row per teacher, sorted by overall_score descending.

    Example
    -------
    >>> import pandas as pd
    >>> from utils.aggregator import aggregate
    >>> df = pd.read_csv("data/feedback_dataset.csv")
    >>> result = aggregate(df)
    >>> print(result[["teacher_name", "overall_score", "grade", "total_reviews"]])
    """
    cfg = config or _DEFAULT_CONFIG
    cfg.validate()
    scored_df = run_pipeline_on_df(df, cfg)
    return aggregate_from_scores(scored_df, cfg)


# ── Utility: pivot to long format (for radar / bar charts) ────────────────────
def pivot_aspect_scores(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the aggregated DataFrame from wide format to long format
    (teacher × aspect → score), suitable for grouped bar / radar charts.

    Returns
    -------
    pd.DataFrame  columns: teacher_name, aspect, score
    """
    id_cols = ["teacher_name", "overall_score", "grade", "rank"]
    id_cols = [c for c in id_cols if c in agg_df.columns]

    long = agg_df.melt(
        id_vars=id_cols,
        value_vars=SCORE_COLS,
        var_name="aspect_col",
        value_name="score",
    )

    # Human-readable aspect label
    col_to_label = {v: k for k, v in ASPECT_COL.items()}
    long["aspect"] = long["aspect_col"].map(col_to_label)
    return long.drop(columns=["aspect_col"]).reset_index(drop=True)


# ── Utility: single-teacher report dict ──────────────────────────────────────
def teacher_report_dict(
    agg_df: pd.DataFrame,
    teacher_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Extract a single teacher's aggregated row as a plain dict.

    Parameters
    ----------
    agg_df       : output of :func:`aggregate` or :func:`aggregate_from_scores`
    teacher_name : exact teacher name string

    Returns
    -------
    dict or None (if teacher not found)
    """
    row = agg_df[agg_df["teacher_name"] == teacher_name]
    if row.empty:
        logger.warning("Teacher '%s' not found in aggregated DataFrame.", teacher_name)
        return None
    return row.iloc[0].to_dict()


# ── Utility: summary statistics ───────────────────────────────────────────────
def cohort_summary(agg_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute cohort-level statistics across all teachers.

    Returns
    -------
    dict
        n_teachers, mean_overall, std_overall, top_teacher,
        bottom_teacher, high_confidence_count, grade_distribution
    """
    if agg_df.empty:
        return {}

    grade_dist = agg_df["grade"].value_counts().to_dict() if "grade" in agg_df.columns else {}

    return {
        "n_teachers":             int(len(agg_df)),
        "total_reviews":          int(agg_df["total_reviews"].sum()) if "total_reviews" in agg_df.columns else 0,
        "mean_overall":           round(float(agg_df["overall_score"].mean()), 2),
        "std_overall":            round(float(agg_df["overall_score"].std()), 2),
        "median_overall":         round(float(agg_df["overall_score"].median()), 2),
        "top_teacher":            agg_df.iloc[0]["teacher_name"],
        "top_score":              float(agg_df.iloc[0]["overall_score"]),
        "bottom_teacher":         agg_df.iloc[-1]["teacher_name"],
        "bottom_score":           float(agg_df.iloc[-1]["overall_score"]),
        "high_confidence_count":  int((agg_df["confidence_band"] == "High").sum()) if "confidence_band" in agg_df.columns else 0,
        "grade_distribution":     grade_dist,
        "best_aspect":            agg_df[SCORE_COLS].mean().idxmax().replace("_score", "").replace("_", " ").title() if all(c in agg_df.columns for c in SCORE_COLS) else None,
        "weakest_aspect":         agg_df[SCORE_COLS].mean().idxmin().replace("_score", "").replace("_", " ").title() if all(c in agg_df.columns for c in SCORE_COLS) else None,
    }
