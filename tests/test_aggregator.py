"""
tests/test_aggregator.py
------------------------
Unit tests for utils/aggregator.py — fully offline (no model inference).

All NLP pipeline calls are mocked so tests run instantly without
downloading any HuggingFace models.

Run with:
    python tests/test_aggregator.py
    pytest tests/test_aggregator.py -v
"""

from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.aggregator import (
    AggregationConfig,
    aggregate_from_scores,
    pivot_aspect_scores,
    teacher_report_dict,
    cohort_summary,
    SCORE_COLS,
    _grade,
    _confidence_band,
    _normalise_to_10,
    _top_n_from_lists,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
def _make_scored_df(n_teachers: int = 3, reviews_each: int = 10, seed: int = 0) -> pd.DataFrame:
    """
    Build a synthetic pre-scored DataFrame with realistic per-row NLP outputs.
    No real pipeline is called.
    """
    rng = np.random.default_rng(seed)
    teachers = [f"Teacher_{chr(65 + i)}" for i in range(n_teachers)]
    subjects  = ["Maths", "Physics", "Chemistry", "Biology", "History",
                 "English", "CS", "Economics"][:n_teachers]

    rows: List[Dict[str, Any]] = []
    for t, subj in zip(teachers, subjects):
        for _ in range(reviews_each):
            comm  = float(rng.uniform(4.0, 9.5))
            sk    = float(rng.uniform(4.0, 9.5))
            eng   = float(rng.uniform(3.5, 9.0))
            resp  = float(rng.uniform(3.5, 9.0))
            aq    = float(rng.uniform(3.0, 8.5))
            conf  = float(rng.uniform(0.55, 0.95))
            label = rng.choice(["Positive", "Neutral", "Negative"],
                               p=[0.5, 0.3, 0.2])
            rows.append({
                "teacher_name":             t,
                "subject":                  subj,
                "feedback_text":            f"Feedback for {t}.",
                "communication_score":      round(comm, 4),
                "subject_knowledge_score":  round(sk,   4),
                "engagement_score":         round(eng,  4),
                "responsiveness_score":     round(resp, 4),
                "assignment_quality_score": round(aq,   4),
                "overall_score":            round((comm + sk + eng + resp + aq) / 5, 4),
                "confidence":               round(conf, 4),
                "sentiment_label":          label,
                "is_mixed":                 bool(rng.random() < 0.15),
                "strengths_list":           ["clear explanations"] if rng.random() > 0.5 else [],
                "improvements_list":        ["assignment workload"] if rng.random() > 0.6 else [],
            })

    return pd.DataFrame(rows)


def _minimal_scored_df() -> pd.DataFrame:
    """Two teachers, 5 reviews each — deterministic values."""
    rows = []
    for teacher, comm, sk, eng, resp, aq, conf, label in [
        # Teacher_X — consistently high scores
        ("Teacher_X", 9.0, 8.8, 8.5, 8.0, 7.5, 0.90, "Positive"),
        ("Teacher_X", 8.5, 9.0, 8.0, 8.5, 7.8, 0.88, "Positive"),
        ("Teacher_X", 9.2, 8.5, 9.0, 7.8, 8.0, 0.92, "Positive"),
        ("Teacher_X", 8.8, 9.1, 8.3, 8.2, 7.6, 0.89, "Positive"),
        ("Teacher_X", 9.1, 8.7, 8.7, 8.0, 7.9, 0.91, "Positive"),
        # Teacher_Y — consistently lower scores
        ("Teacher_Y", 5.5, 5.0, 4.8, 4.5, 4.0, 0.65, "Negative"),
        ("Teacher_Y", 5.0, 5.5, 5.0, 4.8, 4.5, 0.62, "Neutral"),
        ("Teacher_Y", 5.8, 4.8, 4.5, 5.0, 4.2, 0.67, "Negative"),
        ("Teacher_Y", 5.2, 5.2, 5.2, 4.6, 4.3, 0.63, "Neutral"),
        ("Teacher_Y", 5.4, 5.1, 4.9, 4.7, 4.1, 0.64, "Negative"),
    ]:
        rows.append({
            "teacher_name": teacher, "subject": "Test",
            "feedback_text": "feedback",
            "communication_score": comm, "subject_knowledge_score": sk,
            "engagement_score": eng, "responsiveness_score": resp,
            "assignment_quality_score": aq, "overall_score": comm,
            "confidence": conf, "sentiment_label": label, "is_mixed": False,
            "strengths_list": ["clear explanations"] if label == "Positive" else [],
            "improvements_list": ["improve engagement"] if label == "Negative" else [],
        })
    return pd.DataFrame(rows)


# ── Helper ────────────────────────────────────────────────────────────────────
REQUIRED_OUTPUT_COLS = {
    "teacher_name", "total_reviews",
    "communication_score", "subject_knowledge_score",
    "engagement_score", "responsiveness_score", "assignment_quality_score",
    "overall_score", "confidence_score", "confidence_band",
    "grade", "descriptor",
    "positive_pct", "neutral_pct", "negative_pct", "mixed_count",
    "strengths", "improvements", "rank",
}


def _check_output_schema(df: pd.DataFrame, label: str = "") -> List[str]:
    errors: List[str] = []
    prefix = f"[{label}] " if label else ""
    missing = REQUIRED_OUTPUT_COLS - set(df.columns)
    if missing:
        errors.append(f"{prefix}Missing columns: {missing}")
    for col in SCORE_COLS + ["overall_score"]:
        if col in df.columns:
            bad = df[(df[col] < 0) | (df[col] > 10)]
            if not bad.empty:
                errors.append(f"{prefix}{col} out of [0,10]: {bad[col].tolist()}")
    if "confidence_score" in df.columns:
        bad = df[(df["confidence_score"] < 0) | (df["confidence_score"] > 1)]
        if not bad.empty:
            errors.append(f"{prefix}confidence_score out of [0,1]: {bad['confidence_score'].tolist()}")
    if "rank" in df.columns:
        expected_ranks = set(range(1, len(df) + 1))
        actual_ranks   = set(df["rank"].tolist())
        if expected_ranks != actual_ranks:
            errors.append(f"{prefix}rank values wrong: {actual_ranks}")
    return errors


# ── Tests ──────────────────────────────────────────────────────────────────────
def test_output_schema_basic():
    """aggregate_from_scores() must return all required columns."""
    df  = _make_scored_df()
    agg = aggregate_from_scores(df)
    errors = _check_output_schema(agg, "basic")
    assert not errors, "\n".join(errors)
    print("PASS  test_output_schema_basic")


def test_one_row_per_teacher():
    """Output must have exactly one row per unique teacher."""
    df  = _make_scored_df(n_teachers=5, reviews_each=8)
    agg = aggregate_from_scores(df)
    assert len(agg) == 5, f"Expected 5 rows, got {len(agg)}"
    assert agg["teacher_name"].nunique() == 5
    print(f"PASS  test_one_row_per_teacher  ({len(agg)} teachers)")


def test_total_reviews_correct():
    """total_reviews must equal the actual row count per teacher."""
    reviews_each = 12
    df  = _make_scored_df(n_teachers=3, reviews_each=reviews_each)
    agg = aggregate_from_scores(df)
    for _, row in agg.iterrows():
        assert row["total_reviews"] == reviews_each, (
            f"{row['teacher_name']}: expected {reviews_each}, got {row['total_reviews']}"
        )
    print(f"PASS  test_total_reviews_correct  ({reviews_each} reviews/teacher)")


def test_score_columns_in_range():
    """All 5 aspect scores and overall_score must be in [0, 10]."""
    df  = _make_scored_df(n_teachers=4, reviews_each=15)
    agg = aggregate_from_scores(df)
    errors = _check_output_schema(agg, "range")
    assert not errors, "\n".join(errors)
    print("PASS  test_score_columns_in_range")


def test_weighted_formula():
    """
    With a custom config (all weights to Communication),
    overall_score should equal communication_score.
    """
    all_comm_cfg = AggregationConfig(weights={
        "Communication":      1.0,
        "Subject Knowledge":  0.0,
        "Engagement":         0.0,
        "Responsiveness":     0.0,
        "Assignment Quality": 0.0,
    })
    df  = _make_scored_df(n_teachers=2, reviews_each=10, seed=1)
    agg = aggregate_from_scores(df, config=all_comm_cfg)
    for _, row in agg.iterrows():
        assert abs(row["overall_score"] - row["communication_score"]) < 0.01, (
            f"overall={row['overall_score']} != comm={row['communication_score']}"
        )
    print("PASS  test_weighted_formula  (all weight on communication)")


def test_default_weights_sum_to_one():
    """Default AggregationConfig weights must sum exactly to 1.0."""
    cfg   = AggregationConfig()
    total = sum(cfg.weights.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}"
    # Verify the specific required weights
    assert cfg.weights["Communication"]      == 0.25
    assert cfg.weights["Subject Knowledge"]  == 0.25
    assert cfg.weights["Engagement"]         == 0.20
    assert cfg.weights["Responsiveness"]     == 0.15
    assert cfg.weights["Assignment Quality"] == 0.15
    print("PASS  test_default_weights_sum_to_one  (25/25/20/15/15)")


def test_rank_ordering():
    """Teachers must be ranked so rank=1 has the highest overall_score."""
    df  = _make_scored_df(n_teachers=5, reviews_each=10)
    agg = aggregate_from_scores(df)
    scores = agg["overall_score"].tolist()
    assert scores == sorted(scores, reverse=True), (
        f"Scores not sorted descending: {scores}"
    )
    assert agg.iloc[0]["rank"] == 1
    assert agg.iloc[-1]["rank"] == len(agg)
    print(f"PASS  test_rank_ordering  (top={agg.iloc[0]['teacher_name']}, "
          f"score={agg.iloc[0]['overall_score']})")


def test_high_scorer_ranks_first():
    """Teacher_X (high scores) must rank above Teacher_Y (low scores)."""
    df  = _minimal_scored_df()
    agg = aggregate_from_scores(df)
    assert agg.iloc[0]["teacher_name"] == "Teacher_X", (
        f"Expected Teacher_X at rank 1, got {agg.iloc[0]['teacher_name']}"
    )
    assert agg.iloc[1]["teacher_name"] == "Teacher_Y"
    assert agg.iloc[0]["overall_score"] > agg.iloc[1]["overall_score"]
    print(f"PASS  test_high_scorer_ranks_first  "
          f"(X={agg.iloc[0]['overall_score']:.2f}, Y={agg.iloc[1]['overall_score']:.2f})")


def test_confidence_score_range():
    """confidence_score must be in [0, 1] for all teachers."""
    df  = _make_scored_df(n_teachers=4, reviews_each=20)
    agg = aggregate_from_scores(df)
    bad = agg[(agg["confidence_score"] < 0) | (agg["confidence_score"] > 1)]
    assert bad.empty, f"confidence_score out of range:\n{bad[['teacher_name','confidence_score']]}"
    print("PASS  test_confidence_score_range")


def test_confidence_band_values():
    """confidence_band must only be 'High', 'Medium', or 'Low'."""
    df  = _make_scored_df(n_teachers=3, reviews_each=5)
    agg = aggregate_from_scores(df)
    valid = {"High", "Medium", "Low"}
    bad   = set(agg["confidence_band"].tolist()) - valid
    assert not bad, f"Invalid confidence_band values: {bad}"
    print(f"PASS  test_confidence_band_values  "
          f"({agg['confidence_band'].value_counts().to_dict()})")


def test_high_volume_raises_confidence():
    """
    A teacher with many reviews should have a higher confidence_score
    than one with very few, given similar per-review model confidence.
    """
    cfg = AggregationConfig()

    def _make_teacher(n: int, conf_val: float = 0.80) -> pd.DataFrame:
        return pd.DataFrame({
            "teacher_name": ["T"] * n,
            "communication_score": [7.0] * n,
            "subject_knowledge_score": [7.0] * n,
            "engagement_score": [7.0] * n,
            "responsiveness_score": [7.0] * n,
            "assignment_quality_score": [7.0] * n,
            "confidence": [conf_val] * n,
            "sentiment_label": ["Positive"] * n,
            "is_mixed": [False] * n,
        })

    few  = aggregate_from_scores(_make_teacher(2), cfg)
    many = aggregate_from_scores(_make_teacher(20), cfg)
    assert many.iloc[0]["confidence_score"] >= few.iloc[0]["confidence_score"], (
        f"More reviews should give >= confidence: "
        f"few={few.iloc[0]['confidence_score']}, many={many.iloc[0]['confidence_score']}"
    )
    print(f"PASS  test_high_volume_raises_confidence  "
          f"(n=2 -> {few.iloc[0]['confidence_score']:.3f}, "
          f"n=20 -> {many.iloc[0]['confidence_score']:.3f})")


def test_grade_assignment():
    """Grade must match expected thresholds."""
    cases = [
        (9.5, "A+", "Outstanding"),
        (8.2, "A",  "Excellent"),
        (7.1, "B",  "Good"),
        (6.3, "C",  "Satisfactory"),
        (5.1, "D",  "Needs Improvement"),
        (3.0, "F",  "Poor"),
    ]
    for score, expected_grade, expected_desc in cases:
        g, d = _grade(score)
        assert g == expected_grade, f"score={score}: expected {expected_grade}, got {g}"
        assert d == expected_desc,  f"score={score}: expected {expected_desc}, got {d}"
    print("PASS  test_grade_assignment  (6 threshold cases)")


def test_sentiment_percentages_sum_to_100():
    """positive_pct + neutral_pct + negative_pct must sum to 100."""
    df  = _make_scored_df(n_teachers=3, reviews_each=20)
    agg = aggregate_from_scores(df)
    for _, row in agg.iterrows():
        total = row["positive_pct"] + row["neutral_pct"] + row["negative_pct"]
        assert abs(total - 100.0) < 0.5, (
            f"{row['teacher_name']}: sentiment pcts sum to {total}"
        )
    print("PASS  test_sentiment_percentages_sum_to_100")


def test_normalisation_to_10_scale():
    """After cross-teacher normalisation, max aspect score must be 10.0."""
    df  = _make_scored_df(n_teachers=5, reviews_each=10, seed=99)
    agg = aggregate_from_scores(df)
    for col in SCORE_COLS:
        max_val = agg[col].max()
        assert abs(max_val - 10.0) < 0.05, (
            f"{col}: max after normalisation is {max_val:.3f}, expected ≈10.0"
        )
    print("PASS  test_normalisation_to_10_scale")


def test_single_teacher():
    """Single-teacher input must still return one valid row."""
    df = pd.DataFrame([{
        "teacher_name": "Solo Teacher",
        "feedback_text": "Great teacher.",
        "communication_score": 8.0,
        "subject_knowledge_score": 7.5,
        "engagement_score": 7.0,
        "responsiveness_score": 6.5,
        "assignment_quality_score": 6.0,
        "confidence": 0.80,
        "sentiment_label": "Positive",
        "is_mixed": False,
        "strengths_list": ["clear explanations"],
        "improvements_list": [],
    }])
    agg = aggregate_from_scores(df)
    assert len(agg) == 1
    errors = _check_output_schema(agg, "single_teacher")
    assert not errors, "\n".join(errors)
    assert agg.iloc[0]["rank"] == 1
    print("PASS  test_single_teacher")


def test_missing_score_columns_filled():
    """Missing score columns must be filled with 5.0 without raising errors."""
    df = pd.DataFrame([
        {"teacher_name": "T1", "communication_score": 7.0, "confidence": 0.75,
         "sentiment_label": "Positive", "is_mixed": False},
        {"teacher_name": "T1", "communication_score": 7.5, "confidence": 0.78,
         "sentiment_label": "Positive", "is_mixed": False},
    ])
    agg = aggregate_from_scores(df)   # missing subject_knowledge, engagement, etc.
    assert len(agg) == 1
    assert "overall_score" in agg.columns
    print("PASS  test_missing_score_columns_filled")


def test_no_teacher_name_column():
    """DataFrame without teacher_name must treat all rows as one teacher."""
    df = _make_scored_df(n_teachers=1, reviews_each=8)
    df = df.drop(columns=["teacher_name"])
    agg = aggregate_from_scores(df)
    assert len(agg) == 1
    assert agg.iloc[0]["teacher_name"] == "Unknown Teacher"
    print("PASS  test_no_teacher_name_column")


def test_pivot_aspect_scores():
    """pivot_aspect_scores must return n_teachers × 5 rows."""
    df  = _make_scored_df(n_teachers=3, reviews_each=10)
    agg = aggregate_from_scores(df)
    long = pivot_aspect_scores(agg)
    assert len(long) == 3 * 5, f"Expected 15 rows, got {len(long)}"
    assert "aspect" in long.columns
    assert "score"  in long.columns
    assert set(long["aspect"].unique()) == {
        "Communication", "Subject Knowledge",
        "Engagement", "Responsiveness", "Assignment Quality"
    }
    print(f"PASS  test_pivot_aspect_scores  ({len(long)} rows)")


def test_teacher_report_dict():
    """teacher_report_dict must return a dict with all required keys."""
    df  = _make_scored_df(n_teachers=3, reviews_each=10)
    agg = aggregate_from_scores(df)
    name = agg.iloc[0]["teacher_name"]
    report = teacher_report_dict(agg, name)
    assert report is not None
    assert report["teacher_name"] == name
    for col in ["overall_score", "grade", "total_reviews", "confidence_score"]:
        assert col in report, f"Missing key in report dict: {col}"
    print(f"PASS  test_teacher_report_dict  (teacher={name})")


def test_teacher_report_dict_not_found():
    """teacher_report_dict must return None for unknown teacher names."""
    df  = _make_scored_df(n_teachers=2, reviews_each=5)
    agg = aggregate_from_scores(df)
    result = teacher_report_dict(agg, "Nonexistent Teacher")
    assert result is None
    print("PASS  test_teacher_report_dict_not_found")


def test_cohort_summary():
    """cohort_summary must return a dict with expected keys."""
    n_t = 4
    n_r = 10
    df  = _make_scored_df(n_teachers=n_t, reviews_each=n_r)
    agg = aggregate_from_scores(df)
    summary = cohort_summary(agg)
    required = {"n_teachers", "total_reviews", "mean_overall", "std_overall",
                "median_overall", "top_teacher", "bottom_teacher",
                "grade_distribution", "best_aspect", "weakest_aspect"}
    missing = required - set(summary.keys())
    assert not missing, f"Missing keys in cohort_summary: {missing}"
    assert summary["n_teachers"]   == n_t,       f"Expected {n_t}, got {summary['n_teachers']}"
    assert summary["total_reviews"] == n_t * n_r, f"Expected {n_t*n_r}, got {summary['total_reviews']}"
    assert summary["top_teacher"] != summary["bottom_teacher"]
    print(f"PASS  test_cohort_summary  "
          f"(top={summary['top_teacher']}, mean={summary['mean_overall']:.2f})")


def test_custom_weights_validated():
    """AggregationConfig.validate() must raise on weights != 1.0."""
    bad_cfg = AggregationConfig(weights={
        "Communication":      0.30,  # sums to 1.05
        "Subject Knowledge":  0.25,
        "Engagement":         0.20,
        "Responsiveness":     0.15,
        "Assignment Quality": 0.15,
    })
    try:
        bad_cfg.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "1.0" in str(e)
    print("PASS  test_custom_weights_validated")


def test_json_serializable():
    """All aggregated DataFrame values must be JSON-serializable."""
    df  = _make_scored_df(n_teachers=3, reviews_each=8)
    agg = aggregate_from_scores(df)
    try:
        agg.to_json(orient="records")
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"Output is not JSON-serializable: {exc}") from exc
    print("PASS  test_json_serializable")


def test_normalise_to_10_helper():
    """_normalise_to_10 must map series to [0, 10] with max = 10."""
    series = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    result = _normalise_to_10(series)
    assert abs(result.max() - 10.0) < 1e-6
    assert abs(result.min() -  0.0) < 1e-6
    # Constant series → clipped unchanged
    const = _normalise_to_10(pd.Series([5.0, 5.0, 5.0]))
    assert const.tolist() == [5.0, 5.0, 5.0]
    print("PASS  test_normalise_to_10_helper")


def test_top_n_from_lists():
    """_top_n_from_lists must return top-N most frequent items."""
    series = pd.Series([
        ["clear explanations", "high engagement"],
        ["clear explanations"],
        ["clear explanations", "deep knowledge"],
        ["high engagement"],
    ])
    result = _top_n_from_lists(series, n=2)
    items = [x.strip() for x in result.split("|")]
    assert items[0] == "clear explanations", f"Most frequent should be first: {items}"
    assert len(items) <= 2
    print(f"PASS  test_top_n_from_lists  (result='{result}')")


def test_empty_dataframe():
    """Empty DataFrame input must return empty DataFrame without error."""
    df  = pd.DataFrame(columns=["teacher_name", "feedback_text"] + SCORE_COLS + ["confidence"])
    agg = aggregate_from_scores(df)
    assert isinstance(agg, pd.DataFrame)
    assert len(agg) == 0
    print("PASS  test_empty_dataframe")


# ── Mock test: aggregate() with pipeline patch ────────────────────────────────
def test_aggregate_with_mocked_pipeline():
    """
    aggregate() end-to-end test with the NLP pipeline mocked out.
    Verifies the full pipeline → aggregation flow without model inference.
    """
    mock_result = {
        "overall_score": 7.5,
        "communication_score": 8.0,
        "subject_knowledge_score": 7.8,
        "engagement_score": 7.2,
        "responsiveness_score": 6.9,
        "assignment_quality_score": 6.5,
        "confidence": 0.83,
        "sentiment_label": "Positive",
        "is_mixed": False,
        "strengths": ["clear explanations"],
        "improvements": ["assignment workload"],
        "_text": "mock",
        "_sentiment_probs": {"Positive": 0.83, "Neutral": 0.10, "Negative": 0.07},
        "_aspect_detail": {},
        "_is_duplicate": False,
    }

    raw_df = pd.DataFrame([
        {"teacher_name": "Dr. Alpha", "feedback_text": "Great teacher.", "subject": "Maths"},
        {"teacher_name": "Dr. Alpha", "feedback_text": "Very knowledgeable.", "subject": "Maths"},
        {"teacher_name": "Dr. Beta",  "feedback_text": "Average class.",    "subject": "Physics"},
        {"teacher_name": "Dr. Beta",  "feedback_text": "Could improve.",    "subject": "Physics"},
        {"teacher_name": "Dr. Beta",  "feedback_text": "Needs more interaction.", "subject": "Physics"},
    ])

    with patch("nlp.pipeline.analyze_batch", return_value=[mock_result] * len(raw_df)):
        from utils.aggregator import aggregate
        agg = aggregate(raw_df)

    assert len(agg) == 2, f"Expected 2 teachers, got {len(agg)}"
    assert set(agg["teacher_name"]) == {"Dr. Alpha", "Dr. Beta"}

    beta_row = agg[agg["teacher_name"] == "Dr. Beta"].iloc[0]
    assert beta_row["total_reviews"] == 3

    errors = _check_output_schema(agg, "mocked_pipeline")
    assert not errors, "\n".join(errors)
    print(f"PASS  test_aggregate_with_mocked_pipeline  "
          f"({len(agg)} teachers, schema valid)")


# ── Runner ────────────────────────────────────────────────────────────────────
ALL_TESTS = [
    test_output_schema_basic,
    test_one_row_per_teacher,
    test_total_reviews_correct,
    test_score_columns_in_range,
    test_weighted_formula,
    test_default_weights_sum_to_one,
    test_rank_ordering,
    test_high_scorer_ranks_first,
    test_confidence_score_range,
    test_confidence_band_values,
    test_high_volume_raises_confidence,
    test_grade_assignment,
    test_sentiment_percentages_sum_to_100,
    test_normalisation_to_10_scale,
    test_single_teacher,
    test_missing_score_columns_filled,
    test_no_teacher_name_column,
    test_pivot_aspect_scores,
    test_teacher_report_dict,
    test_teacher_report_dict_not_found,
    test_cohort_summary,
    test_custom_weights_validated,
    test_json_serializable,
    test_normalise_to_10_helper,
    test_top_n_from_lists,
    test_empty_dataframe,
    test_aggregate_with_mocked_pipeline,
]


def run_all() -> bool:
    print("\n" + "=" * 65)
    print("   Score Aggregation Module — Test Suite")
    print("=" * 65 + "\n")
    passed = failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {test_fn.__name__}")
            print(f"      {exc}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 65}")
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)} tests")
    print("=" * 65 + "\n")
    return failed == 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    success = run_all()
    sys.exit(0 if success else 1)
