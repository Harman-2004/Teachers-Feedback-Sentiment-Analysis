"""
nlp/pipeline.py
---------------
Teacher Feedback NLP Pipeline — main orchestration module.

Exposes two entry points:

    analyze(text)          → single feedback dict (canonical JSON)
    analyze_batch(texts)   → list[dict] (deduplication + batch processing)

Canonical output schema
-----------------------
{
    "overall_score":            7.8,   # 0–10
    "communication_score":      9.1,
    "subject_knowledge_score":  8.5,
    "engagement_score":         7.4,
    "responsiveness_score":     7.0,
    "assignment_quality_score": 4.8,
    "strengths":  ["clear explanations", "deep subject knowledge"],
    "improvements": ["assignment workload", "improve response time"],
    "confidence": 0.89,
    "sentiment_label": "Positive",
    "is_mixed": False,
    # extended fields (always present):
    "_text":           "<original input>",
    "_sentiment_probs": {"Positive": 0.91, "Neutral": 0.06, "Negative": 0.03},
    "_aspect_detail": {
        "Communication":      {"score_10": 9.1, "relevance": 0.62, "detected": True},
        "Subject Knowledge":  {"score_10": 8.5, ...},
        ...
    },
    "_is_duplicate": False
}

Usage
-----
    from nlp.pipeline import analyze, analyze_batch

    # Single
    result = analyze("The teacher explains concepts clearly but assignments are excessive.")
    print(result)

    # Batch
    results = analyze_batch([
        "Excellent teacher, very engaging and knowledgeable.",
        "Lectures are boring and feedback is always delayed.",
        "Excellent teacher, very engaging and knowledgeable.",   # duplicate → reused
    ])
    for r in results:
        print(r["overall_score"], r["_is_duplicate"])

Configuration
-------------
    from nlp.pipeline import PipelineConfig, analyze
    cfg = PipelineConfig(sentiment_model="distilbert-base-uncased-finetuned-sst-2-english",
                         encoder_model="all-MiniLM-L6-v2",
                         batch_size=32)
    result = analyze("…", config=cfg)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Lazy imports so the module can be imported before torch is ready ──────────
def _get_sentiment():
    from nlp.sentiment import analyze_sentiment
    return analyze_sentiment

def _get_aspects():
    from nlp.aspects import score_aspects_batch
    return score_aspects_batch

def _get_scorer():
    from nlp.scoring import score_batch
    return score_batch

def _get_str_imp():
    from nlp.summarizer import extract_strengths, extract_improvements
    return extract_strengths, extract_improvements


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """
    Configures which models and options the pipeline uses.

    Attributes
    ----------
    sentiment_model  : HuggingFace sentiment model ID
    encoder_model    : SentenceTransformer model ID
    batch_size       : texts per inference batch
    deduplicate      : skip identical texts, reuse cached results
    sentence_level   : run ABSA per sentence (more sensitive, slower)
    """
    sentiment_model : str  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    encoder_model   : str  = "all-MiniLM-L6-v2"
    batch_size      : int  = 16
    deduplicate     : bool = True
    sentence_level  : bool = True


_DEFAULT_CONFIG = PipelineConfig()


# ── Duplicate detection ───────────────────────────────────────────────────────
def _fingerprint(text: str) -> str:
    """SHA-256 hex digest of stripped, lower-cased text."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


# ── Core pipeline ─────────────────────────────────────────────────────────────
def _run_pipeline(
    texts: List[str],
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    """
    Internal: run full NLP pipeline on a *deduplicated* list of texts.

    Returns one result dict per input text (order preserved).
    """
    if not texts:
        return []

    t0 = time.perf_counter()

    # 1. Sentiment
    analyze_sentiment = _get_sentiment()
    sent_results = analyze_sentiment(
        texts,
        model_name=config.sentiment_model,
        batch_size=config.batch_size,
        deduplicate=False,          # already deduplicated upstream
    )
    logger.debug("Sentiment done in %.2fs", time.perf_counter() - t0)

    # 2. Aspect scoring
    t1 = time.perf_counter()
    score_aspects_batch = _get_aspects()
    asp_results = score_aspects_batch(
        texts,
        model_name=config.encoder_model,
        deduplicate=False,
        sentence_level=config.sentence_level,
    )
    logger.debug("Aspects done in %.2fs", time.perf_counter() - t1)

    # 3. Composite scoring → canonical JSON
    t2 = time.perf_counter()
    score_batch = _get_scorer()
    scored = score_batch(texts, sent_results, asp_results)
    logger.debug("Scoring done in %.2fs", time.perf_counter() - t2)

    # 4. Attach extended metadata
    extract_strengths, extract_improvements = _get_str_imp()
    output: List[Dict[str, Any]] = []
    for text, sent_r, asp_wrap, score_r in zip(
        texts, sent_results, asp_results, scored
    ):
        # Merge strength/improvement from summarizer (richer phrase patterns)
        extra_str = extract_strengths(text)
        extra_imp = extract_improvements(text)

        # Union-merge with scoring engine's keyword signals
        merged_str = _dedupe_list(score_r.get("strengths", []) + extra_str)[:5]
        merged_imp = _dedupe_list(score_r.get("improvements", []) + extra_imp)[:5]

        result = {
            # ── Canonical fields ──────────────────────────────────────────
            "overall_score":            score_r["overall_score"],
            "communication_score":      score_r["communication_score"],
            "subject_knowledge_score":  score_r["subject_knowledge_score"],
            "engagement_score":         score_r["engagement_score"],
            "responsiveness_score":     score_r["responsiveness_score"],
            "assignment_quality_score": score_r["assignment_quality_score"],
            "strengths":                merged_str,
            "improvements":             merged_imp,
            "confidence":               score_r["confidence"],
            "sentiment_label":          score_r["sentiment_label"],
            "is_mixed":                 score_r["is_mixed"],
            # ── Extended metadata ─────────────────────────────────────────
            "_text":             text,
            "_sentiment_probs":  sent_r.get("probs", {}),
            "_aspect_detail": {
                asp: {
                    "score_10":    info.get("score_10",    5.0),
                    "relevance":   info.get("relevance",   0.0),
                    "detected":    info.get("detected",    False),
                    "positive_sim":info.get("positive_sim",0.0),
                    "negative_sim":info.get("negative_sim",0.0),
                }
                for asp, info in asp_wrap.get("aspects", {}).items()
            },
            "_is_duplicate": False,
        }
        output.append(result)

    logger.info(
        "Pipeline complete: %d texts in %.2fs",
        len(texts),
        time.perf_counter() - t0,
    )
    return output


def _dedupe_list(lst: List[str]) -> List[str]:
    """Preserve order while removing duplicates."""
    seen: set = set()
    result: List[str] = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ── Public API ────────────────────────────────────────────────────────────────
def analyze(
    text: str,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Run the full NLP pipeline on a **single** feedback string.

    Parameters
    ----------
    text   : raw feedback string
    config : optional PipelineConfig (uses defaults if omitted)

    Returns
    -------
    dict — canonical pipeline JSON (see module docstring for schema)

    Example
    -------
    >>> from nlp.pipeline import analyze
    >>> r = analyze("The teacher explains topics brilliantly but gives too much homework.")
    >>> print(r["overall_score"], r["strengths"], r["improvements"])
    """
    cfg = config or _DEFAULT_CONFIG
    if not text or not text.strip():
        return _empty_result(text)

    results = _run_pipeline([text.strip()], cfg)
    return results[0]


def analyze_batch(
    texts: List[str],
    config: Optional[PipelineConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full NLP pipeline on a **list** of feedback strings.

    Features
    --------
    • Deduplication: identical texts are analysed once; duplicates receive
      the cached result with ``_is_duplicate = True``.
    • Batch inference: sentiment and aspect models process texts in batches
      for efficiency.
    • Order preserved: output list matches the input list 1-to-1.

    Parameters
    ----------
    texts  : list of raw feedback strings (may contain duplicates)
    config : optional PipelineConfig

    Returns
    -------
    list[dict] — one canonical dict per input text

    Example
    -------
    >>> results = analyze_batch(["Great teacher!", "Great teacher!", "Bad class."])
    >>> results[0]["_is_duplicate"]  # False
    >>> results[1]["_is_duplicate"]  # True
    """
    cfg = config or _DEFAULT_CONFIG

    if not texts:
        return []

    # ── Deduplication ────────────────────────────────────────────────────────
    fp_to_text: Dict[str, str] = {}
    unique_texts: List[str] = []
    input_fps: List[str] = []

    for raw in texts:
        cleaned = (raw or "").strip()
        fp = _fingerprint(cleaned) if cleaned else "EMPTY"
        input_fps.append(fp)
        if cleaned and fp not in fp_to_text:
            fp_to_text[fp] = cleaned
            unique_texts.append(cleaned)

    # ── Run pipeline on unique texts ─────────────────────────────────────────
    unique_results: Dict[str, Dict[str, Any]] = {}

    if unique_texts:
        pipeline_results = _run_pipeline(unique_texts, cfg)
        for text, result in zip(unique_texts, pipeline_results):
            fp = _fingerprint(text)
            unique_results[fp] = result

    # ── Re-expand to original order ──────────────────────────────────────────
    final: List[Dict[str, Any]] = []
    first_seen: Dict[str, bool] = {}
    for i, fp in enumerate(input_fps):
        if fp in unique_results:
            r = dict(unique_results[fp])        # shallow copy
            is_dup = fp in first_seen
            first_seen.setdefault(fp, True)
            r["_is_duplicate"] = is_dup
            r["_text"] = (texts[i] or "").strip()
            final.append(r)
        else:
            final.append(_empty_result((texts[i] or "").strip()))

    return final


def _empty_result(text: str = "") -> Dict[str, Any]:
    """Return a zeroed-out canonical result for empty / invalid input."""
    return {
        "overall_score":            0.0,
        "communication_score":      0.0,
        "subject_knowledge_score":  0.0,
        "engagement_score":         0.0,
        "responsiveness_score":     0.0,
        "assignment_quality_score": 0.0,
        "strengths":                [],
        "improvements":             [],
        "confidence":               0.0,
        "sentiment_label":          "Neutral",
        "is_mixed":                 False,
        "_text":                    text,
        "_sentiment_probs":         {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0},
        "_aspect_detail":           {},
        "_is_duplicate":            False,
    }


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    DEMO_TEXTS = [
        # Mixed — strong communication, weak assignments
        "The teacher explains concepts with exceptional clarity and uses excellent "
        "analogies. However, the assignment workload is excessive and deadlines are "
        "unrealistically tight.",

        # Positive across all dimensions
        "An outstanding educator. Responds to every email within hours, keeps content "
        "updated with the latest research, and makes every class genuinely interactive.",

        # Negative — poor engagement and responsiveness
        "Classes are purely lecture-based with zero student interaction. The teacher "
        "never replies to emails and office hours are rarely held.",

        # Duplicate of first entry
        "The teacher explains concepts with exceptional clarity and uses excellent "
        "analogies. However, the assignment workload is excessive and deadlines are "
        "unrealistically tight.",

        # Short neutral
        "Adequate course delivery. Nothing exceptional but no major issues either.",
    ]

    if len(sys.argv) > 1:
        # Accept a single text from the command line
        user_text = " ".join(sys.argv[1:])
        print("\nAnalysing single input...\n")
        r = analyze(user_text)
        print(json.dumps(r, indent=2))
    else:
        print("\nRunning demo batch analysis...\n")
        results = analyze_batch(DEMO_TEXTS)
        for i, r in enumerate(results):
            dup_tag = " [DUPLICATE]" if r["_is_duplicate"] else ""
            print(f"--- Entry {i + 1}{dup_tag} ---")
            print(f"  Text snippet   : {r['_text'][:80]}…")
            print(f"  overall_score  : {r['overall_score']}")
            print(f"  communication  : {r['communication_score']}")
            print(f"  subject_know   : {r['subject_knowledge_score']}")
            print(f"  engagement     : {r['engagement_score']}")
            print(f"  responsiveness : {r['responsiveness_score']}")
            print(f"  assignment_q   : {r['assignment_quality_score']}")
            print(f"  strengths      : {r['strengths']}")
            print(f"  improvements   : {r['improvements']}")
            print(f"  confidence     : {r['confidence']}")
            print(f"  sentiment      : {r['sentiment_label']}  (mixed={r['is_mixed']})")
            print()
