"""
tests/test_report_generator.py
-------------------------------
Offline unit tests for utils/report_generator.py.
"""

from __future__ import annotations

import io
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.report_generator import generate_csv_export, generate_summary_csv_report


def test_generate_csv_export():
    df = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    csv_bytes = generate_csv_export(df)
    assert isinstance(csv_bytes, bytes)
    
    # Read it back
    df_read = pd.read_csv(io.BytesIO(csv_bytes))
    assert list(df_read.columns) == ["col1", "col2"]
    assert df_read.shape == (2, 2)


def test_generate_summary_csv_report_basic():
    # Setup mock scored_df
    df = pd.DataFrame({
        "teacher_name": ["Teacher X", "Teacher X"],
        "subject": ["Maths", "Maths"],
        "feedback_text": ["Great explanations.", "Really responsive."],
        "communication_score": [8.5, 9.0],
        "subject_knowledge_score": [9.0, 9.0],
        "engagement_score": [8.0, 8.0],
        "responsiveness_score": [8.5, 8.5],
        "assignment_quality_score": [7.0, 7.0],
        "confidence": [0.9, 0.9],
        "sentiment_label": ["Positive", "Positive"]
    })
    
    mock_nlp = {
        "summarize_teacher_feedback": lambda comments, scores: {
            "summary": "Mock summary for Teacher X",
            "strengths": ["clear explanations"],
            "improvements": ["none"]
        }
    }
    
    csv_bytes = generate_summary_csv_report(
        df_filtered=df,
        nlp_modules=mock_nlp,
        run_summarizer=True,
        cache_dict={}
    )
    
    assert isinstance(csv_bytes, bytes)
    df_report = pd.read_csv(io.BytesIO(csv_bytes))
    
    # Check expected columns
    expected_cols = [
        "Teacher Name", "Subject", "Total Reviews", "Overall Score (0-100)",
        "Grade", "Descriptor", "Communication Score", "Subject Knowledge Score",
        "Engagement Score", "Responsiveness Score", "Assignment Quality Score",
        "Confidence Score", "Confidence Band", "AI Summary", "AI Strengths", "AI Improvements"
    ]
    for col in expected_cols:
        assert col in df_report.columns
        
    assert len(df_report) == 1
    assert df_report.iloc[0]["Teacher Name"] == "Teacher X"
    assert df_report.iloc[0]["AI Summary"] == "Mock summary for Teacher X"
    assert df_report.iloc[0]["AI Strengths"] == "clear explanations"


def test_generate_summary_csv_report_caching():
    df = pd.DataFrame({
        "teacher_name": ["Teacher X"],
        "feedback_text": ["Great explanations."],
        "communication_score": [8.5],
        "subject_knowledge_score": [9.0],
        "engagement_score": [8.0],
        "responsiveness_score": [8.5],
        "assignment_quality_score": [7.0],
        "confidence": [0.9],
        "sentiment_label": ["Positive"]
    })
    
    # Pre-populate cache dict
    import hashlib
    comments_hash = hashlib.sha256(b"Great explanations.").hexdigest()
    cache_key = f"Teacher X_{comments_hash}"
    
    cache = {
        cache_key: {
            "summary": "Cached summary here",
            "strengths": ["cached strength"],
            "improvements": ["cached improvement"]
        }
    }
    
    # Mock nlp should NOT be called since it is cached
    mock_nlp = {
        "summarize_teacher_feedback": lambda comments, scores: Exception("Should not be called")
    }
    
    csv_bytes = generate_summary_csv_report(
        df_filtered=df,
        nlp_modules=mock_nlp,
        run_summarizer=True,
        cache_dict=cache
    )
    
    df_report = pd.read_csv(io.BytesIO(csv_bytes))
    assert df_report.iloc[0]["AI Summary"] == "Cached summary here"
    assert df_report.iloc[0]["AI Strengths"] == "cached strength"


if __name__ == "__main__":
    test_generate_csv_export()
    test_generate_summary_csv_report_basic()
    test_generate_summary_csv_report_caching()
    print("ALL TESTS PASSED")
