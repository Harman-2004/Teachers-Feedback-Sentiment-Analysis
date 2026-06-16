"""
nlp package — Teacher Feedback Analytics NLP Pipeline

Quick start
-----------
    from nlp.pipeline import analyze, analyze_batch

    result = analyze("The teacher is excellent but assignments are excessive.")
    results = analyze_batch(["feedback 1", "feedback 2", "feedback 1"])  # deduplicates
"""

from nlp.pipeline import analyze, analyze_batch, PipelineConfig
from nlp.summarizer import summarize_teacher_feedback

__all__ = ["analyze", "analyze_batch", "PipelineConfig", "summarize_teacher_feedback"]
