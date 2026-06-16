"""
tests/test_teacher_summarizer.py
--------------------------------
Unit tests for the teacher feedback summarization module.
Runs fully offline with mocked transformers components.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.summarizer import (
    summarize_teacher_feedback,
    _anonymize_student_mentions,
)

# ── Mocking Data ─────────────────────────────────────────────────────────────
FEW_COMMENTS = [
    "Student A says she is very clear in communication.",
    "I enjoyed the lectures, they were very structured.",
    "My class felt that responsiveness could be faster.",
]

MANY_COMMENTS = [
    "Student A says she is very clear in communication.",
    "I enjoyed the lectures, they were very structured.",
    "My class felt that responsiveness could be faster.",
    "One student noted that grading is inconsistent.",
    "Another student commented that the subject depth is amazing.",
    "I think she is a great instructor overall.",
    "She replies slowly to emails and requests.",
]

ASPECT_SCORES = {
    "Communication": 8.5,
    "Subject Knowledge": 9.0,
    "Engagement": 7.0,
    "Responsiveness": 4.5,
    "Assignment Quality": 5.0,
}

# ── Tests ──────────────────────────────────────────────────────────────────────

def test_anonymize_student_mentions():
    """Verify that student labels, names, and first-person pronouns are filtered out."""
    assert "feedback" in _anonymize_student_mentions("Student A said she was good.").lower()
    assert "feedback" in _anonymize_student_mentions("Student X is very helpful.").lower()
    assert "it was noted that" in _anonymize_student_mentions("I feel that the lectures are confusing.").lower()
    assert "feedback appreciated" in _anonymize_student_mentions("I loved the assignment topics.").lower()
    assert "the class" in _anonymize_student_mentions("My students like the slides.").lower()
    print("PASS  test_anonymize_student_mentions")


def test_low_volume_warning_trigger():
    """If comments < 5, summary must start with the limited sample prefix."""
    with patch("nlp.summarizer._load_summarizer", return_value=None):
        res = summarize_teacher_feedback(FEW_COMMENTS, ASPECT_SCORES)
        summary = res["summary"]
        assert summary.startswith("Based on a limited sample of 3 reviews, feedback notes that:")
        assert "Student A" not in summary
        assert "I enjoyed" not in summary
        print("PASS  test_low_volume_warning_trigger")


def test_word_count_limit():
    """Summary must be strictly under 80 words."""
    # Extractive fallback
    with patch("nlp.summarizer._load_summarizer", return_value=None):
        res = summarize_teacher_feedback(MANY_COMMENTS, ASPECT_SCORES)
        summary = res["summary"]
        word_count = len(summary.split())
        assert word_count < 80, f"Summary has {word_count} words: {summary}"
        
    # Abstractive path with mock model returning a long string
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"summary_text": "This is a very long summary " * 30}]
    with patch("nlp.summarizer._load_summarizer", return_value=mock_pipe):
        res = summarize_teacher_feedback(MANY_COMMENTS, ASPECT_SCORES)
        summary = res["summary"]
        word_count = len(summary.split())
        assert word_count < 80, f"Summary has {word_count} words: {summary}"
    print("PASS  test_word_count_limit")


def test_strengths_and_improvements_aligned_with_aspects():
    """Verify strengths align with high aspect scores and improvements with low aspect scores."""
    with patch("nlp.summarizer._load_summarizer", return_value=None):
        res = summarize_teacher_feedback(MANY_COMMENTS, ASPECT_SCORES)
        # Communication (8.5) and Subject Knowledge (9.0) are highest -> strengths should align with these
        # Responsiveness (4.5) and Assignment Quality (5.0) are lowest -> improvements should align with these
        strengths = res["strengths"]
        improvements = res["improvements"]
        
        # Check that Communication/Subject Knowledge keywords show up in strengths
        assert any("explanations" in s or "knowledge" in s or "lectures" in s for s in strengths)
        # Check that Responsiveness/Assignment Quality keywords show up in improvements
        assert any("response" in i or "workload" in i or "grading" in i for i in improvements)
        print("PASS  test_strengths_and_improvements_aligned_with_aspects")


def test_extractive_fallback_correctness():
    """Verify extractive fallback generates a cohesive summary from strong/weak aspects."""
    with patch("nlp.summarizer._load_summarizer", return_value=None):
        res = summarize_teacher_feedback(MANY_COMMENTS, ASPECT_SCORES)
        summary = res["summary"]
        assert len(summary) > 0
        assert "Student" not in summary
        assert "I " not in summary
        print("PASS  test_extractive_fallback_correctness")


def test_empty_input_handling():
    """Verify empty comments or score dictionary returns clean empty state."""
    res = summarize_teacher_feedback([], ASPECT_SCORES)
    assert res["summary"] == "No feedback comments available."
    assert res["strengths"] == []
    assert res["improvements"] == []
    print("PASS  test_empty_input_handling")


def test_anonymize_no_crash_on_complex_text():
    """Verify anonymization works without throwing regex errors on complex texts."""
    complex_text = "I think... student A, student B, student X, my students, we were happy. I liked the class."
    anon = _anonymize_student_mentions(complex_text)
    assert "student A" not in anon.lower()
    assert "my students" not in anon.lower()
    assert "I " not in anon
    print("PASS  test_anonymize_no_crash_on_complex_text")


def test_aspect_scores_scale_handling():
    """Verify it correctly normalizes aspect scores in 0-100 range down to 0-10."""
    high_scale_scores = {
        "Communication": 85.0,
        "Subject Knowledge": 90.0,
        "Engagement": 70.0,
        "Responsiveness": 45.0,
        "Assignment Quality": 50.0,
    }
    with patch("nlp.summarizer._load_summarizer", return_value=None):
        res = summarize_teacher_feedback(MANY_COMMENTS, high_scale_scores)
        # If it normalized, strengths and improvements will align properly
        assert len(res["strengths"]) > 0
        assert len(res["improvements"]) > 0
    print("PASS  test_aspect_scores_scale_handling")


# ── Runner ────────────────────────────────────────────────────────────────────
ALL_TESTS = [
    test_anonymize_student_mentions,
    test_low_volume_warning_trigger,
    test_word_count_limit,
    test_strengths_and_improvements_aligned_with_aspects,
    test_extractive_fallback_correctness,
    test_empty_input_handling,
    test_anonymize_no_crash_on_complex_text,
    test_aspect_scores_scale_handling,
]

def run_all() -> bool:
    print("\n" + "=" * 65)
    print("   Teacher Feedback Summarizer — Test Suite")
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
