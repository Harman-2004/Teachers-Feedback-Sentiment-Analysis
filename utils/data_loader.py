"""
utils/data_loader.py
--------------------
Data ingestion utilities for teacher feedback datasets.

Supports:
  - CSV upload
  - Excel (.xlsx) upload
  - Manual text entry (single feedback)
  - Sample dataset generation for demo purposes

Expected CSV/Excel columns (flexible mapping supported):
  - feedback_text  : the raw feedback string  (required)
  - teacher_name   : teacher identifier        (optional)
  - date           : submission date           (optional)
  - subject        : course / subject taught   (optional)
  - rating         : numeric rating (1–5)      (optional)
"""

from __future__ import annotations

import io
import logging
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column aliases — maps common column name variants to canonical names
# ---------------------------------------------------------------------------
COLUMN_ALIASES: Dict[str, List[str]] = {
    "feedback_text": [
        "feedback", "comment", "comments", "review", "reviews",
        "text", "feedback_text", "student_feedback", "response",
        "review_text", "review text", "feedback text", "comments_text",
        "comments text", "comment_text", "comment text", "response_text",
        "response text", "reviews_text", "reviews text", "student_comment",
        "student_comments", "student_review", "student_reviews", "evaluation",
        "evaluations",
    ],
    "teacher_name": [
        "teacher", "teacher_name", "instructor", "professor",
        "faculty", "lecturer", "name", "teacher name", "instructor name",
        "professor name", "faculty name", "lecturer name", "staff name",
        "staff", "educator", "educator name",
    ],
    "date": [
        "date", "submission_date", "created_at", "timestamp", "submitted_on",
        "submission date", "submitted date", "date submitted", "created date",
        "time", "datetime",
    ],
    "subject": [
        "subject", "course", "class", "module", "department", "subject name",
        "course name", "class name", "module name", "course code", "course_code",
    ],
    "rating": [
        "rating", "score", "stars", "grade_given", "marks", "overall_rating",
        "overall rating", "teacher rating", "teacher_rating", "rating score",
        "score rating", "stars rating",
    ],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using alias mapping."""
    rename_map: Dict[str, str] = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols and canonical not in df.columns:
                rename_map[lower_cols[alias]] = canonical
                break
    return df.rename(columns=rename_map)


def load_csv(file_obj) -> pd.DataFrame:
    """
    Load feedback data from a CSV file object (e.g., Streamlit UploadedFile).

    Returns
    -------
    pd.DataFrame with normalized column names.
    """
    try:
        df = pd.read_csv(file_obj, encoding="utf-8")
    except UnicodeDecodeError:
        file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding="latin-1")
    df = _normalize_columns(df)
    df = _clean_dataframe(df)
    logger.info(f"Loaded CSV: {len(df)} rows, columns={list(df.columns)}")
    return df


def load_excel(file_obj) -> pd.DataFrame:
    """
    Load feedback data from an Excel file object.

    Returns
    -------
    pd.DataFrame with normalized column names.
    """
    df = pd.read_excel(file_obj, engine="openpyxl")
    df = _normalize_columns(df)
    df = _clean_dataframe(df)
    logger.info(f"Loaded Excel: {len(df)} rows, columns={list(df.columns)}")
    return df


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows and ensure feedback_text column exists."""
    df = df.dropna(how="all")

    if "feedback_text" not in df.columns:
        # Attempt to use first string column as feedback text
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if str_cols:
            df = df.rename(columns={str_cols[0]: "feedback_text"})
            logger.warning(
                f"No 'feedback_text' column found. Using '{str_cols[0]}' as feedback."
            )
        else:
            raise ValueError(
                "Dataset must contain a text column for feedback. "
                "Expected column name: 'feedback', 'comment', 'review', or 'text'."
            )

    df["feedback_text"] = df["feedback_text"].astype(str).str.strip()
    df = df[df["feedback_text"].str.len() > 5]

    # Parse dates if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Normalize rating to 0–100 if present
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        max_r = df["rating"].max()
        if max_r and max_r <= 10:
            df["rating_normalized"] = (df["rating"] / max_r * 100).round(1)
        elif max_r and max_r <= 5:
            df["rating_normalized"] = (df["rating"] / 5 * 100).round(1)
        else:
            df["rating_normalized"] = df["rating"]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sample dataset generator
# ---------------------------------------------------------------------------
SAMPLE_TEACHERS = [
    "Dr. Aisha Khan",
    "Prof. Ravi Sharma",
    "Ms. Priya Nair",
    "Mr. James Okafor",
    "Dr. Emily Chen",
]

SAMPLE_SUBJECTS = [
    "Mathematics", "Physics", "Computer Science",
    "English Literature", "Chemistry",
]

POSITIVE_FEEDBACK = [
    "An exceptional teacher who explains concepts with remarkable clarity and patience.",
    "Very engaging and knowledgeable. Makes even difficult topics easy to understand.",
    "Always available for doubt-clearing sessions. Truly passionate about teaching.",
    "The best teacher I've had. Innovative teaching methods and great communication skills.",
    "Provides excellent real-world examples and keeps the class very interactive.",
    "Highly professional and punctual. Course content is well-structured and relevant.",
    "Great at making students feel comfortable to ask questions. Very supportive.",
    "Thoroughly prepares lessons and gives constructive feedback on assignments.",
    "Dynamic and enthusiastic. Classes are never boring, always something new to learn.",
    "Subject knowledge is outstanding. Explains the 'why' behind every concept.",
]

NEUTRAL_FEEDBACK = [
    "The teacher covers the syllabus adequately. Could use more practical examples.",
    "Teaching is satisfactory. Some topics could be explained in more depth.",
    "Classes are on time and content is standard. Nothing exceptional but no issues.",
    "Good knowledge of the subject but needs to improve student interaction.",
    "Assignments are fair. The teaching pace is sometimes too fast.",
    "Average experience. The teacher is available but not very proactive.",
    "Lectures are informative but could be more engaging for students.",
    "The course content is relevant though the delivery can be monotonous at times.",
]

NEGATIVE_FEEDBACK = [
    "Very difficult to understand due to fast speaking pace and unclear explanations.",
    "Often arrives late to class which is very disrespectful of students' time.",
    "The feedback on assignments is vague and not helpful for improvement.",
    "Does not engage with students. Lectures are one-sided and boring.",
    "Limited knowledge beyond the textbook. Cannot answer deeper questions.",
    "Very poor classroom management. Class is frequently chaotic and unproductive.",
    "Course content feels outdated and not aligned with current industry standards.",
    "The teacher is unapproachable and dismissive when students ask for help.",
]


def generate_sample_dataset(
    n_rows: int = 150,
    n_teachers: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic sample dataset for demonstration purposes.

    Parameters
    ----------
    n_rows : int
        Number of feedback entries to generate.
    n_teachers : int
        Number of teachers (capped at len(SAMPLE_TEACHERS)).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
    """
    random.seed(seed)
    np.random.seed(seed)

    teachers = SAMPLE_TEACHERS[:min(n_teachers, len(SAMPLE_TEACHERS))]
    subjects = SAMPLE_SUBJECTS[:min(n_teachers, len(SAMPLE_SUBJECTS))]
    teacher_subject_map = dict(zip(teachers, subjects))

    # Weighted feedback distribution: 50% pos, 30% neutral, 20% neg
    all_feedback = (
        POSITIVE_FEEDBACK * 5 + NEUTRAL_FEEDBACK * 3 + NEGATIVE_FEEDBACK * 2
    )

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    rows = []
    for _ in range(n_rows):
        teacher = random.choice(teachers)
        feedback = random.choice(all_feedback)
        rating_base = (
            random.uniform(3.5, 5.0) if feedback in POSITIVE_FEEDBACK
            else random.uniform(2.5, 3.5) if feedback in NEUTRAL_FEEDBACK
            else random.uniform(1.0, 2.5)
        )
        date = start_date + timedelta(days=random.randint(0, date_range))
        rows.append({
            "feedback_text": feedback,
            "teacher_name": teacher,
            "subject": teacher_subject_map[teacher],
            "rating": round(rating_base, 1),
            "date": date.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(rows)
    return _clean_dataframe(df)


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a loaded dataset and return diagnostic information.

    Returns
    -------
    dict
        - ``is_valid``         : bool
        - ``n_rows``           : int
        - ``has_teacher_col``  : bool
        - ``has_date_col``     : bool
        - ``has_rating_col``   : bool
        - ``missing_pct``      : float (% of empty feedback cells)
        - ``warnings``         : list[str]
    """
    warnings: List[str] = []
    is_valid = True

    if "feedback_text" not in df.columns:
        is_valid = False
        warnings.append("Missing required column: feedback_text")

    missing_pct = 0.0
    if "feedback_text" in df.columns:
        missing = df["feedback_text"].isna().sum() + (df["feedback_text"] == "").sum()
        missing_pct = round(missing / len(df) * 100, 2) if len(df) > 0 else 0.0
        if missing_pct > 20:
            warnings.append(f"{missing_pct}% of feedback entries are empty or missing.")

    has_teacher = "teacher_name" in df.columns
    has_date = "date" in df.columns
    has_rating = "rating" in df.columns

    if not has_teacher:
        warnings.append("No 'teacher_name' column found — analysis will be aggregated.")
    if not has_date:
        warnings.append("No 'date' column found — trend analysis will be unavailable.")

    return {
        "is_valid": is_valid,
        "n_rows": len(df),
        "has_teacher_col": has_teacher,
        "has_date_col": has_date,
        "has_rating_col": has_rating,
        "missing_pct": missing_pct,
        "warnings": warnings,
    }
