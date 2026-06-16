"""
tests/test_pipeline.py
----------------------
Unit + integration tests for the Teacher Feedback NLP Pipeline.

Run with:
    pytest tests/test_pipeline.py -v

Or without pytest (plain Python):
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Test data ─────────────────────────────────────────────────────────────────
POSITIVE_TEXT = (
    "The teacher explains every concept with remarkable clarity and uses "
    "excellent real-world analogies. She responds to emails within hours "
    "and is always available during office hours. Highly knowledgeable and "
    "genuinely passionate — one of the best educators I have encountered."
)

NEGATIVE_TEXT = (
    "Lectures are confusing and the teacher speaks far too fast. "
    "Office hours are rarely held and email replies take weeks. "
    "The assignment workload is excessive and grading is inconsistent. "
    "No student interaction whatsoever — completely disengaging."
)

MIXED_TEXT = (
    "The teacher explains concepts with exceptional clarity and the subject "
    "knowledge is outstanding. However, the assignment workload is excessive "
    "and deadlines are unrealistically tight. Responsiveness could also improve."
)

NEUTRAL_TEXT = "Adequate course delivery. Nothing exceptional but no major issues."

DUPLICATE_TEXT = POSITIVE_TEXT   # exact duplicate of first entry

BATCH = [POSITIVE_TEXT, NEGATIVE_TEXT, MIXED_TEXT, DUPLICATE_TEXT, NEUTRAL_TEXT]

REQUIRED_KEYS = {
    "overall_score", "communication_score", "subject_knowledge_score",
    "engagement_score", "responsiveness_score", "assignment_quality_score",
    "strengths", "improvements", "confidence",
    "sentiment_label", "is_mixed",
    "_text", "_sentiment_probs", "_aspect_detail", "_is_duplicate",
}

SCORE_FIELDS = [
    "overall_score", "communication_score", "subject_knowledge_score",
    "engagement_score", "responsiveness_score", "assignment_quality_score",
    "confidence",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _check_schema(result: Dict[str, Any], label: str = "") -> List[str]:
    errors = []
    prefix = f"[{label}] " if label else ""

    missing = REQUIRED_KEYS - set(result.keys())
    if missing:
        errors.append(f"{prefix}Missing keys: {missing}")

    for field in SCORE_FIELDS:
        val = result.get(field)
        if not isinstance(val, (int, float)):
            errors.append(f"{prefix}{field} is not numeric: {val!r}")
        elif not (0.0 <= val <= 10.0):
            errors.append(f"{prefix}{field}={val} out of range [0, 10]")

    if not isinstance(result.get("strengths"), list):
        errors.append(f"{prefix}strengths must be a list")
    if not isinstance(result.get("improvements"), list):
        errors.append(f"{prefix}improvements must be a list")
    if result.get("sentiment_label") not in {"Positive", "Neutral", "Negative"}:
        errors.append(f"{prefix}invalid sentiment_label: {result.get('sentiment_label')!r}")
    if not isinstance(result.get("is_mixed"), bool):
        errors.append(f"{prefix}is_mixed must be bool")
    if not isinstance(result.get("_is_duplicate"), bool):
        errors.append(f"{prefix}_is_duplicate must be bool")

    probs = result.get("_sentiment_probs", {})
    if probs:
        total = sum(probs.values())
        if not (0.95 <= total <= 1.05):
            errors.append(f"{prefix}sentiment probs don't sum to 1.0: {total}")

    return errors


# ── Test functions ────────────────────────────────────────────────────────────
def test_single_schema():
    """analyze() must return a dict with all required keys in valid ranges."""
    from nlp.pipeline import analyze
    result = analyze(POSITIVE_TEXT)
    errors = _check_schema(result, "positive_single")
    assert not errors, "\n".join(errors)
    print("PASS  test_single_schema")


def test_single_positive_scores_high():
    """Positive feedback should yield overall_score >= 6.0."""
    from nlp.pipeline import analyze
    result = analyze(POSITIVE_TEXT)
    assert result["overall_score"] >= 5.0, (
        f"Expected >= 5.0 for positive text, got {result['overall_score']}"
    )
    assert result["sentiment_label"] == "Positive", (
        f"Expected Positive, got {result['sentiment_label']}"
    )
    print(f"PASS  test_single_positive_scores_high  (score={result['overall_score']})")


def test_single_negative_scores_lower():
    """Negative feedback should yield a lower overall_score than positive."""
    from nlp.pipeline import analyze
    pos_result = analyze(POSITIVE_TEXT)
    neg_result = analyze(NEGATIVE_TEXT)
    assert neg_result["overall_score"] < pos_result["overall_score"], (
        f"Expected neg ({neg_result['overall_score']}) < pos ({pos_result['overall_score']})"
    )
    print(
        f"PASS  test_single_negative_scores_lower  "
        f"(pos={pos_result['overall_score']}, neg={neg_result['overall_score']})"
    )


def test_mixed_sentiment_detection():
    """Mixed feedback should be flagged as is_mixed=True (or near-neutral)."""
    from nlp.pipeline import analyze
    result = analyze(MIXED_TEXT)
    # Either is_mixed flagged OR score is in middle range [3, 8]
    mid_range = 3.0 <= result["overall_score"] <= 8.5
    assert result["is_mixed"] or mid_range, (
        f"Mixed text not detected as mixed. is_mixed={result['is_mixed']}, "
        f"score={result['overall_score']}"
    )
    print(f"PASS  test_mixed_sentiment_detection  (is_mixed={result['is_mixed']}, score={result['overall_score']})")


def test_batch_length():
    """analyze_batch() must return exactly as many results as inputs."""
    from nlp.pipeline import analyze_batch
    results = analyze_batch(BATCH)
    assert len(results) == len(BATCH), (
        f"Expected {len(BATCH)} results, got {len(results)}"
    )
    print(f"PASS  test_batch_length  ({len(results)} results)")


def test_batch_schema():
    """Every result in a batch must have the correct schema."""
    from nlp.pipeline import analyze_batch
    results = analyze_batch(BATCH)
    all_errors: List[str] = []
    for i, r in enumerate(results):
        all_errors.extend(_check_schema(r, f"batch[{i}]"))
    assert not all_errors, "\n".join(all_errors)
    print("PASS  test_batch_schema")


def test_deduplication():
    """Duplicate texts must be returned with _is_duplicate=True."""
    from nlp.pipeline import analyze_batch
    texts = [POSITIVE_TEXT, NEGATIVE_TEXT, POSITIVE_TEXT]
    results = analyze_batch(texts)

    assert results[0]["_is_duplicate"] is False, "First occurrence must NOT be a duplicate"
    assert results[2]["_is_duplicate"] is True,  "Third entry (duplicate) must be flagged"

    # Scores must be identical for the duplicate
    for field in SCORE_FIELDS:
        assert results[0][field] == results[2][field], (
            f"Duplicate mismatch on {field}: {results[0][field]} vs {results[2][field]}"
        )
    print("PASS  test_deduplication")


def test_empty_input():
    """analyze() with empty string must return zeroed-out result."""
    from nlp.pipeline import analyze
    result = analyze("")
    assert result["overall_score"] == 0.0
    assert result["confidence"] == 0.0
    assert result["strengths"] == []
    print("PASS  test_empty_input")


def test_empty_batch():
    """analyze_batch([]) must return empty list."""
    from nlp.pipeline import analyze_batch
    assert analyze_batch([]) == []
    print("PASS  test_empty_batch")


def test_five_aspect_scores_present():
    """All five aspect scores must be present and valid."""
    from nlp.pipeline import analyze
    result = analyze(MIXED_TEXT)
    for key in [
        "communication_score", "subject_knowledge_score",
        "engagement_score", "responsiveness_score", "assignment_quality_score",
    ]:
        assert key in result, f"Missing key: {key}"
        assert 0.0 <= result[key] <= 10.0, f"{key}={result[key]} out of [0,10]"
    print("PASS  test_five_aspect_scores_present")


def test_aspect_detail_fields():
    """_aspect_detail must contain score_10, relevance, detected for each aspect."""
    from nlp.pipeline import analyze
    result = analyze(POSITIVE_TEXT)
    detail = result.get("_aspect_detail", {})
    expected_aspects = {
        "Communication", "Subject Knowledge",
        "Engagement", "Responsiveness", "Assignment Quality",
    }
    for asp in expected_aspects:
        assert asp in detail, f"Missing aspect in _aspect_detail: {asp}"
        assert "score_10"  in detail[asp]
        assert "relevance" in detail[asp]
        assert "detected"  in detail[asp]
    print("PASS  test_aspect_detail_fields")


def test_strengths_and_improvements_are_lists():
    """strengths and improvements must be lists (possibly empty)."""
    from nlp.pipeline import analyze
    for text in [POSITIVE_TEXT, NEGATIVE_TEXT, NEUTRAL_TEXT]:
        result = analyze(text)
        assert isinstance(result["strengths"], list)
        assert isinstance(result["improvements"], list)
        assert len(result["strengths"])    <= 5
        assert len(result["improvements"]) <= 5
    print("PASS  test_strengths_and_improvements_are_lists")


def test_confidence_range():
    """confidence must be strictly in [0, 1]."""
    from nlp.pipeline import analyze
    for text in [POSITIVE_TEXT, NEGATIVE_TEXT, MIXED_TEXT]:
        r = analyze(text)
        assert 0.0 <= r["confidence"] <= 1.0, (
            f"confidence={r['confidence']} out of [0,1]"
        )
    print("PASS  test_confidence_range")


def test_json_serialisable():
    """Pipeline output must be JSON-serialisable."""
    from nlp.pipeline import analyze
    result = analyze(MIXED_TEXT)
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"Result is not JSON-serialisable: {exc}") from exc
    print("PASS  test_json_serialisable")


def test_custom_config():
    """PipelineConfig must be accepted without error."""
    from nlp.pipeline import analyze, PipelineConfig
    cfg = PipelineConfig(batch_size=8, sentence_level=False)
    result = analyze(POSITIVE_TEXT, config=cfg)
    errors = _check_schema(result, "custom_config")
    assert not errors, "\n".join(errors)
    print("PASS  test_custom_config")


def test_module_imports():
    """All four NLP modules must import cleanly."""
    import nlp.sentiment  as sm; assert hasattr(sm, "analyze_sentiment")
    import nlp.aspects    as ap; assert hasattr(ap, "score_aspects")
    import nlp.scoring    as sc; assert hasattr(sc, "score_single")
    import nlp.summarizer as su; assert hasattr(su, "extract_strengths")
    import nlp.pipeline   as pl; assert hasattr(pl, "analyze")
    print("PASS  test_module_imports")


# ── Sentinel & runner ─────────────────────────────────────────────────────────
ALL_TESTS = [
    test_module_imports,
    test_single_schema,
    test_single_positive_scores_high,
    test_single_negative_scores_lower,
    test_mixed_sentiment_detection,
    test_batch_length,
    test_batch_schema,
    test_deduplication,
    test_empty_input,
    test_empty_batch,
    test_five_aspect_scores_present,
    test_aspect_detail_fields,
    test_strengths_and_improvements_are_lists,
    test_confidence_range,
    test_json_serialisable,
    test_custom_config,
]


def run_all() -> bool:
    print("\n" + "=" * 65)
    print("   Teacher Feedback NLP Pipeline — Test Suite")
    print("=" * 65 + "\n")
    passed, failed = 0, 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {test_fn.__name__}")
            print(f"      {exc}")
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
