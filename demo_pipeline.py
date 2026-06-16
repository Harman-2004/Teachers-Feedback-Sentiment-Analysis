"""
demo_pipeline.py
----------------
Interactive demo of the Teacher Feedback NLP Pipeline.

Shows the exact canonical JSON output for a set of representative
feedback texts and prints a formatted score card.

Run:
    python demo_pipeline.py
    python demo_pipeline.py "Your custom feedback text here"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)


DEMO_INPUTS = [
    {
        "label": "Highly Positive",
        "text": (
            "An exceptional teacher who explains every concept with remarkable clarity "
            "and patience. Always responds to student emails within hours and holds extra "
            "office hours before exams. The assignments are thoughtfully designed with clear "
            "rubrics and the grading is fair and transparent. Genuinely passionate about the "
            "subject — one of the best educators I have encountered."
        ),
    },
    {
        "label": "Highly Negative",
        "text": (
            "Lectures are completely confusing. The teacher speaks far too fast, mumbles, "
            "and jumps between topics with no transitions. Takes over two weeks to reply to "
            "emails, and office hours are rarely held. The assignment workload is excessive, "
            "grading is inconsistent, and feedback is generic copy-paste. No student "
            "interaction whatsoever — deeply demoralizing."
        ),
    },
    {
        "label": "Mixed (Good Communication, Poor Assignments)",
        "text": (
            "The teacher explains concepts with exceptional clarity and uses excellent analogies. "
            "Subject knowledge is deep and current. However, the assignment workload is excessive "
            "and deadlines are unrealistically tight. Feedback on submissions is generic and "
            "offers no real direction for improvement."
        ),
    },
    {
        "label": "Neutral / Average",
        "text": (
            "Adequate course delivery. The teacher covers the syllabus satisfactorily. "
            "Communication is fine though explanations could be more structured. "
            "Nothing exceptional but no major issues either."
        ),
    },
]


def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for a 0–10 score."""
    filled = int(round(score / 10 * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {score:.1f}"


def _colour_score(score: float) -> str:
    if score >= 7.0:
        return f"\033[92m{score:.2f}\033[0m"   # green
    elif score >= 5.0:
        return f"\033[93m{score:.2f}\033[0m"   # yellow
    else:
        return f"\033[91m{score:.2f}\033[0m"   # red


def print_result(label: str, result: dict) -> None:
    dup = " [DUPLICATE]" if result.get("_is_duplicate") else ""
    mixed = "  (MIXED)" if result.get("is_mixed") else ""
    print(f"\n{'-' * 65}")
    print(f"  {label}{dup}")
    print(f"  Sentiment : {result['sentiment_label']}{mixed}   Confidence: {result['confidence']:.0%}")
    print(f"{'-' * 65}")
    print(f"  {'OVERALL':<28} {_bar(result['overall_score'])}")
    print(f"  {'Communication':<28} {_bar(result['communication_score'])}")
    print(f"  {'Subject Knowledge':<28} {_bar(result['subject_knowledge_score'])}")
    print(f"  {'Engagement':<28} {_bar(result['engagement_score'])}")
    print(f"  {'Responsiveness':<28} {_bar(result['responsiveness_score'])}")
    print(f"  {'Assignment Quality':<28} {_bar(result['assignment_quality_score'])}")
    if result.get("strengths"):
        print(f"\n  Strengths    : {', '.join(result['strengths'])}")
    if result.get("improvements"):
        print(f"  Improvements : {', '.join(result['improvements'])}")


def main() -> None:
    from nlp.pipeline import analyze, analyze_batch

    if len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        print(f"\nAnalysing: \"{custom_text[:80]}...\"" if len(custom_text) > 80 else f"\nAnalysing: \"{custom_text}\"")
        result = analyze(custom_text)
        print_result("Custom Input", result)
        print("\nFull JSON output:")
        print(json.dumps({k: v for k, v in result.items() if not k.startswith("_")}, indent=2))
        return

    print("\n" + "=" * 65)
    print("   Teacher Feedback NLP Pipeline - Demo")
    print("=" * 65)

    texts = [d["text"] for d in DEMO_INPUTS]
    # Add a duplicate to demonstrate deduplication
    texts.append(texts[2])   # duplicate of "Mixed" entry
    labels = [d["label"] for d in DEMO_INPUTS] + ["Mixed (DUPLICATE)"]

    print("\nRunning batch analysis (includes 1 duplicate)...")
    results = analyze_batch(texts)

    for label, result in zip(labels, results):
        print_result(label, result)

    print(f"\n{'=' * 65}")
    print("  Full JSON for entry 1 (Highly Positive):")
    print("=" * 65)
    r = results[0]
    public = {k: v for k, v in r.items() if not k.startswith("_")}
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
