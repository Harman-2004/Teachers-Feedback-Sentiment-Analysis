"""
utils/report_generator.py
--------------------------
PDF and DOCX report generation for teacher performance summaries.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def generate_pdf_report(
    teacher_name: str,
    score_data: Dict[str, Any],
    sentiment_summary_text: str,
    feedback_samples: List[str],
) -> bytes:
    """
    Generate a PDF performance report using fpdf2.

    Returns
    -------
    bytes
        Raw PDF bytes ready for download.
    """
    try:
        from fpdf import FPDF

        class TeacherReport(FPDF):
            def header(self):
                self.set_fill_color(15, 17, 23)
                self.rect(0, 0, 210, 30, "F")
                self.set_font("Helvetica", "B", 18)
                self.set_text_color(230, 230, 255)
                self.cell(0, 15, "Teacher Feedback Analytics", ln=True, align="C")
                self.set_font("Helvetica", "", 10)
                self.set_text_color(148, 163, 184)
                self.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(148, 163, 184)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        pdf = TeacherReport()
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(108, 99, 255)
        pdf.cell(0, 14, teacher_name, ln=True, align="C")
        pdf.ln(4)

        # Score KPIs
        pdf.set_fill_color(26, 29, 46)
        pdf.set_draw_color(42, 45, 62)
        pdf.rect(10, pdf.get_y(), 190, 28, "FD")
        pdf.ln(4)

        score = score_data.get("overall_score", 0)
        grade = score_data.get("grade", "N/A")
        descriptor = score_data.get("descriptor", "")

        pdf.set_font("Helvetica", "B", 28)
        pdf.set_text_color(74, 222, 128)
        pdf.cell(0, 14, f"{score:.1f}/100  ({grade})", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 7, descriptor, ln=True, align="C")
        pdf.ln(8)

        # Sentiment breakdown
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(226, 232, 240)
        pdf.cell(0, 8, "Sentiment Breakdown", ln=True)
        pdf.set_font("Helvetica", "", 11)
        breakdown = score_data.get("sentiment_breakdown", {})
        for label, pct in breakdown.items():
            color = (74, 222, 128) if label == "Positive" else (250, 204, 21) if label == "Neutral" else (248, 113, 113)
            pdf.set_text_color(*color)
            pdf.cell(50, 7, f"  {label}:")
            pdf.set_text_color(226, 232, 240)
            pdf.cell(0, 7, f"{pct:.1f}%", ln=True)
        pdf.ln(5)

        # Aspect scores
        aspect_scores = score_data.get("aspect_scores", {})
        if aspect_scores:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(226, 232, 240)
            pdf.cell(0, 8, "Aspect Performance Scores", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for aspect, sc in sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True):
                bar_width = int(sc * 1.2)  # scale to max 120
                pdf.set_text_color(148, 163, 184)
                pdf.cell(70, 6, f"  {aspect}")
                pdf.set_fill_color(108, 99, 255)
                pdf.rect(pdf.get_x(), pdf.get_y() + 1, bar_width, 4, "F")
                pdf.set_text_color(226, 232, 240)
                pdf.cell(bar_width + 5, 6, "")
                pdf.cell(0, 6, f"{sc:.1f}", ln=True)
            pdf.ln(5)

        # Summary
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(226, 232, 240)
        pdf.cell(0, 8, "AI-Generated Feedback Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(200, 210, 225)
        pdf.multi_cell(0, 6, sentiment_summary_text or "No summary available.")
        pdf.ln(5)

        # Sample feedback
        if feedback_samples:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(226, 232, 240)
            pdf.cell(0, 8, "Sample Feedback Entries", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for i, fb in enumerate(feedback_samples[:5], 1):
                pdf.set_text_color(108, 99, 255)
                pdf.cell(10, 6, f"{i}.")
                pdf.set_text_color(200, 210, 225)
                pdf.multi_cell(0, 5, fb[:200])
                pdf.ln(1)

        return bytes(pdf.output())

    except ImportError:
        logger.error("fpdf2 not installed. Run: pip install fpdf2")
        return b""
    except Exception as exc:
        logger.error(f"PDF generation failed: {exc}")
        return b""


def generate_csv_export(df) -> bytes:
    """Export a DataFrame as UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def generate_summary_csv_report(
    df_filtered: pd.DataFrame,
    nlp_modules: Dict[str, Any],
    run_summarizer: bool = True,
    cache_dict: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generate an aggregated CSV report for all teachers in df_filtered,
    containing overall scores, aspect scores, confidence metrics, and AI summaries.
    """
    import pandas as pd
    import hashlib
    from utils.aggregator import aggregate_from_scores

    agg_df = aggregate_from_scores(df_filtered)
    if agg_df.empty:
        return b""

    records = []
    for _, row in agg_df.iterrows():
        teacher_name = row["teacher_name"]

        # Aspect scores (0-100 scale)
        aspect_scores = {
            "Communication": row["communication_score"] * 10,
            "Subject Knowledge": row["subject_knowledge_score"] * 10,
            "Engagement": row["engagement_score"] * 10,
            "Responsiveness": row["responsiveness_score"] * 10,
            "Assignment Quality": row["assignment_quality_score"] * 10,
        }

        # Get comments for this teacher
        teacher_comments = df_filtered[df_filtered["teacher_name"] == teacher_name]["feedback_text"].tolist()

        summary_text = "Summarization disabled."
        strengths_list = []
        improvements_list = []

        if run_summarizer:
            comments_hash = hashlib.sha256(" ".join(teacher_comments).encode("utf-8")).hexdigest()
            cache_key = f"{teacher_name}_{comments_hash}"

            if cache_dict is not None and cache_key in cache_dict:
                sum_res = cache_dict[cache_key]
                summary_text = sum_res.get("summary", "")
                strengths_list = sum_res.get("strengths", [])
                improvements_list = sum_res.get("improvements", [])
            else:
                if "summarize_teacher_feedback" in nlp_modules:
                    try:
                        sum_res = nlp_modules["summarize_teacher_feedback"](teacher_comments, aspect_scores)
                        summary_text = sum_res.get("summary", "")
                        strengths_list = sum_res.get("strengths", [])
                        improvements_list = sum_res.get("improvements", [])
                        if cache_dict is not None:
                            cache_dict[cache_key] = sum_res
                    except Exception as e:
                        logger.error(f"Error generating summary for {teacher_name}: {e}")
                        summary_text = "Error generating summary."
                else:
                    summary_text = "Summarization module not loaded."

        records.append({
            "Teacher Name": teacher_name,
            "Subject": row.get("subject", "N/A"),
            "Total Reviews": row.get("total_reviews", 0),
            "Overall Score (0-100)": round(row["overall_score"] * 10, 1),
            "Grade": row.get("grade", "N/A"),
            "Descriptor": row.get("descriptor", "N/A"),
            "Communication Score": round(row["communication_score"] * 10, 1),
            "Subject Knowledge Score": round(row["subject_knowledge_score"] * 10, 1),
            "Engagement Score": round(row["engagement_score"] * 10, 1),
            "Responsiveness Score": round(row["responsiveness_score"] * 10, 1),
            "Assignment Quality Score": round(row["assignment_quality_score"] * 10, 1),
            "Confidence Score": round(row.get("confidence_score", 0.0), 2),
            "Confidence Band": row.get("confidence_band", "N/A"),
            "AI Summary": summary_text,
            "AI Strengths": " | ".join(strengths_list),
            "AI Improvements": " | ".join(improvements_list)
        })

    report_df = pd.DataFrame(records)
    return report_df.to_csv(index=False).encode("utf-8")
