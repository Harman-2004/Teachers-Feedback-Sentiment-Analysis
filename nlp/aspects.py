"""
nlp/aspects.py
--------------
Aspect-Based Sentiment Analysis (ABSA) for teacher feedback.

The five pipeline dimensions (matching the required JSON output):
  1. Communication
  2. Subject Knowledge
  3. Engagement
  4. Responsiveness
  5. Assignment Quality

Each dimension has:
  • A rich set of positive / negative / neutral *anchor sentences* encoded
    with SentenceTransformers (all-MiniLM-L6-v2 by default).
  • A SIGNED similarity score computed per sentence chunk of the input text:
      positive anchors contribute +similarity, negative anchors −similarity.
  • The final per-aspect score is mapped to [0.0, 10.0].

Additional public helpers kept for backward compatibility with the dashboard:
  detect_aspects()          — generic top-k aspect detection
  analyze_aspects_batch()   — batch version of detect_aspects
  aspect_frequency_table()  — frequency count helper
"""

from __future__ import annotations

import re
import logging
from functools import lru_cache
from typing import List, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ENCODER = "all-MiniLM-L6-v2"

# ── Five pipeline aspects ────────────────────────────────────────────────────
# Each aspect has "positive" and "negative" anchor sentences.
# The sign of their cosine similarity with the input determines whether the
# aspect sentiment is good or bad.

PIPELINE_ASPECTS: Dict[str, Dict[str, List[str]]] = {
    "Communication": {
        "positive": [
            "The teacher explains concepts with exceptional clarity.",
            "Lectures are well-structured and easy to follow.",
            "Speaks at a perfect, measured pace for note-taking.",
            "Uses simple language and excellent real-world analogies.",
            "Provides clear written feedback on all assignments.",
            "Instructions before each task eliminate confusion entirely.",
            "Always articulate and concise during Q&A sessions.",
            "The teacher narrates examples that make abstract ideas tangible.",
        ],
        "negative": [
            "The teacher speaks far too fast and is hard to follow.",
            "Explanations are unclear and leave students confused.",
            "Instructions are vague and lead to misunderstandings.",
            "Difficult to understand due to poor diction and mumbling.",
            "Jumps between topics without any logical transitions.",
            "Written feedback is one-word and completely unhelpful.",
            "Uses jargon without defining it, creating confusion.",
            "Does not check for understanding before moving on.",
        ],
    },
    "Subject Knowledge": {
        "positive": [
            "The teacher has outstanding depth of subject expertise.",
            "References cutting-edge research and primary sources in lectures.",
            "Can explain any topic ten different ways until understood.",
            "Handles even the most advanced student questions confidently.",
            "Brings personal research into the classroom to enrich lectures.",
            "Keeps the course content updated with latest industry trends.",
            "Every claim is backed by peer-reviewed evidence.",
            "Explains the historical context and evolution of theories.",
        ],
        "negative": [
            "The teacher relies on notes and cannot answer questions confidently.",
            "Made multiple factual errors without acknowledging them.",
            "Course content is outdated and not aligned with modern practice.",
            "Cannot go beyond the textbook when students probe deeper.",
            "Appears to read slides for the first time during lectures.",
            "Foundational theories are explained incorrectly.",
            "The practical examples chosen are technically inaccurate.",
            "Shows limited knowledge beyond what is in the assigned reading.",
        ],
    },
    "Engagement": {
        "positive": [
            "Creates a highly interactive learning environment for students.",
            "Uses gamified quizzes and debates to make learning fun.",
            "Encourages student participation and values every opinion.",
            "The teacher's enthusiasm is contagious and motivates students.",
            "Relates lessons to current events making content immediately relevant.",
            "Makes students feel like co-creators of knowledge, not passive listeners.",
            "Designs collaborative projects that build real skills.",
            "Poses thought-provoking questions that spark genuine curiosity.",
        ],
        "negative": [
            "Classes are purely lecture-based with zero student interaction.",
            "Rarely acknowledges student questions during lectures.",
            "Reading directly from the textbook is completely disengaging.",
            "No group activities, discussions, or real connection with students.",
            "Does not seem interested in teaching; the class feels lifeless.",
            "The classroom feels tense and unwelcoming for participation.",
            "Never poses questions to students, making it easy to zone out.",
            "The same presentation format every week — painfully dull.",
        ],
    },
    "Responsiveness": {
        "positive": [
            "Responds to student emails within hours every time.",
            "Holds extra office hours before exams to support students.",
            "Always follows up on unanswered questions from previous classes.",
            "When students struggle, proactively reaches out to help.",
            "Never dismisses a query; every question receives a thorough answer.",
            "Returns feedback on draft submissions within 48 hours.",
            "Maintains a genuine open-door policy that students actively use.",
            "Creates group chats for real-time doubt clearing outside class.",
        ],
        "negative": [
            "Takes over two weeks to reply to emails with incomplete answers.",
            "Office hours are scheduled but rarely actually held.",
            "Dismisses students who ask for extra support.",
            "Graded assignments are returned weeks after the deadline.",
            "Unavailable outside class and discourages student emails.",
            "Students who visit office hours are told to just read the textbook.",
            "Ignores follow-up questions posted on the course discussion forum.",
            "Never addresses doubt-clearing requests from students.",
        ],
    },
    "Assignment Quality": {
        "positive": [
            "Assignments are thoughtfully designed to bridge theory and practice.",
            "Each task is scaffolded to build progressively toward mastery.",
            "Assignment rubrics are crystal clear about expectations.",
            "Workload is challenging yet fair relative to the credit hours.",
            "Provides detailed model answers after each submission.",
            "Grading is consistent, transparent and never arbitrary.",
            "Assignments encourage independent research and critical thinking.",
            "Group projects are structured for fair contribution from all members.",
        ],
        "negative": [
            "Assignments are poorly worded and require constant clarification.",
            "The workload is excessive for the number of credit hours.",
            "Grading is inconsistent — same quality receives very different marks.",
            "Tasks feel completely disconnected from what is taught in lectures.",
            "Deadlines are unrealistically tight with no flexibility.",
            "No model answers or solutions are provided after submission.",
            "Feedback on assignments is generic and provides no useful guidance.",
            "Many assignments are reused from previous years without update.",
        ],
    },
}

# ── Dashboard backward-compat taxonomy (broader set) ────────────────────────
_COMPAT_ASPECTS: Dict[str, List[str]] = {
    "Teaching Style": [
        "teaching methods", "lecture style", "pedagogy", "teaching approach",
        "explains clearly", "teaching technique", "instructional style",
        "interactive teaching", "boring lectures", "engaging lessons",
    ],
    "Communication": [
        "communication skills", "speaks clearly", "language clarity",
        "verbal communication", "expression", "articulation", "diction",
        "easy to understand", "hard to follow", "clear explanation",
    ],
    "Subject Knowledge": [
        "subject expertise", "knowledge of topic", "domain knowledge",
        "technical knowledge", "deep understanding", "course content mastery",
        "knows the subject", "lacks knowledge", "well-informed teacher",
    ],
    "Classroom Management": [
        "classroom management", "discipline", "controls the class",
        "maintains order", "time management", "class control",
        "organized class", "chaotic classroom", "structured environment",
    ],
    "Student Engagement": [
        "student engagement", "interactive sessions", "student participation",
        "involves students", "motivates students", "encourages questions",
        "active learning", "passive students", "boring class",
    ],
    "Assessment & Feedback": [
        "feedback on assignments", "grading", "assessment",
        "evaluates fairly", "constructive feedback", "marks fairly",
        "exam difficulty", "assignment quality", "homework",
    ],
    "Punctuality & Professionalism": [
        "punctual", "professional", "on time", "respectful",
        "arrives late", "unprofessional", "dedicated teacher",
        "responsible", "attentive", "committed",
    ],
    "Course Content": [
        "syllabus", "course material", "curriculum", "content quality",
        "relevant content", "outdated material", "well-structured course",
        "practical examples", "theoretical content", "real-world application",
    ],
}


# ── Model loading ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_encoder(model_name: str) -> Any:
    import os
    if os.environ.get("FORCE_RULE_BASED") == "1":
        logger.info("Force rule-based mode is active. Skipping SentenceTransformers load.")
        return None
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence encoder: %s", model_name)
        return SentenceTransformer(model_name)
    except (ImportError, OSError, Exception) as exc:
        logger.error("Failed to load sentence encoder: %s. Using rule-based fallback.", exc)
        return None


@lru_cache(maxsize=1)
def _build_pipeline_embeddings(
    model_name: str,
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """
    Pre-encode all positive / negative anchor sentences for each pipeline aspect.
    Returns: {aspect: {"positive": ndarray (N,D), "negative": ndarray (M,D)}}
    """
    enc = _load_encoder(model_name)
    if enc is None:
        return None
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for aspect, poles in PIPELINE_ASPECTS.items():
        result[aspect] = {
            "positive": enc.encode(poles["positive"], convert_to_numpy=True),
            "negative": enc.encode(poles["negative"], convert_to_numpy=True),
        }
    return result


@lru_cache(maxsize=1)
def _build_compat_embeddings(model_name: str) -> Optional[Dict[str, np.ndarray]]:
    """Mean embedding per compat aspect (for detect_aspects / dashboard)."""
    enc = _load_encoder(model_name)
    if enc is None:
        return None
    return {
        asp: enc.encode(phrases, convert_to_numpy=True).mean(axis=0)
        for asp, phrases in _COMPAT_ASPECTS.items()
    }


# ── Internal helpers ─────────────────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    """Split a paragraph into individual sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 8]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _signed_aspect_score(
    text_emb: np.ndarray,
    pos_anchors: np.ndarray,
    neg_anchors: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute signed relevance of a text embedding to an aspect.

    Returns (positive_sim, negative_sim, net_score_0_10)
    """
    # Max-pooled cosine similarity against each pole
    pos_sims = [_cosine_sim(text_emb, a) for a in pos_anchors]
    neg_sims = [_cosine_sim(text_emb, a) for a in neg_anchors]

    pos_max = float(np.mean(sorted(pos_sims, reverse=True)[:3]))  # top-3 mean
    neg_max = float(np.mean(sorted(neg_sims, reverse=True)[:3]))

    # Net score: [-1, +1] → [0, 10]
    net = (pos_max - neg_max + 1.0) / 2.0
    score_10 = round(net * 10.0, 4)
    return round(pos_max, 4), round(neg_max, 4), score_10


# ── Public API ───────────────────────────────────────────────────────────────
def score_aspects(
    text: str,
    model_name: str = DEFAULT_ENCODER,
    sentence_level: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-aspect scores for a single feedback text.

    Parameters
    ----------
    text            : raw feedback string
    model_name      : sentence-transformer model
    sentence_level  : if True, analyse each sentence independently then
                      aggregate — more sensitive to mixed feedback

    Returns
    -------
    dict  keyed by aspect name, each value:
        score_10      float  0.0 – 10.0  (pipeline score)
        positive_sim  float  similarity to positive anchors
        negative_sim  float  similarity to negative anchors
        relevance     float  how strongly the text mentions this aspect
        detected      bool   True when the text clearly references this aspect
    """
    enc = _load_encoder(model_name)
    if enc is None:
        return _score_aspects_rule_based(text)
    embeddings = _build_pipeline_embeddings(model_name)
    if embeddings is None:
        return _score_aspects_rule_based(text)

    if sentence_level:
        sentences = _split_sentences(text)
        if not sentences:
            sentences = [text]
        chunks = sentences
    else:
        chunks = [text]

    chunk_embs = enc.encode(chunks, convert_to_numpy=True)

    aspect_results: Dict[str, Dict[str, Any]] = {}

    for aspect, poles in embeddings.items():
        pos_scores_list: List[float] = []
        neg_scores_list: List[float] = []
        net_scores_list: List[float] = []
        rel_scores_list: List[float] = []

        for chunk_emb in chunk_embs:
            pos_sim, neg_sim, net = _signed_aspect_score(
                chunk_emb, poles["positive"], poles["negative"]
            )
            # relevance = max of both poles (how much the chunk is about this aspect)
            rel = max(pos_sim, neg_sim)
            pos_scores_list.append(pos_sim)
            neg_scores_list.append(neg_sim)
            net_scores_list.append(net)
            rel_scores_list.append(rel)

        # Aggregate: use relevance-weighted average of net scores
        weights = np.array(rel_scores_list, dtype=float)
        w_sum = weights.sum()
        if w_sum > 0:
            weighted_net = float(np.dot(net_scores_list, weights) / w_sum)
        else:
            weighted_net = float(np.mean(net_scores_list))

        relevance = float(np.max(rel_scores_list))
        detected  = relevance > 0.28    # empirical threshold

        aspect_results[aspect] = {
            "score_10":     round(weighted_net, 4),
            "positive_sim": round(float(np.mean(pos_scores_list)), 4),
            "negative_sim": round(float(np.mean(neg_scores_list)), 4),
            "relevance":    round(relevance, 4),
            "detected":     detected,
        }

    return aspect_results


def score_aspects_batch(
    texts: List[str],
    model_name: str = DEFAULT_ENCODER,
    deduplicate: bool = True,
    sentence_level: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run score_aspects on a list of texts (with optional deduplication) using batched encoding.

    Returns
    -------
    list[dict]  — one dict per input text:
        text     str
        aspects  dict  (output of score_aspects)
    """
    enc = _load_encoder(model_name)
    if enc is None:
        unique_results = {t: _score_aspects_rule_based(t) for t in (set(texts) if deduplicate else texts)}
        return [{"text": t, "aspects": unique_results[t]} for t in texts]

    embeddings = _build_pipeline_embeddings(model_name)
    if embeddings is None:
        unique_results = {t: _score_aspects_rule_based(t) for t in (set(texts) if deduplicate else texts)}
        return [{"text": t, "aspects": unique_results[t]} for t in texts]

    if deduplicate:
        unique_texts = []
        seen = set()
        for t in texts:
            if t not in seen:
                seen.add(t)
                unique_texts.append(t)
    else:
        unique_texts = texts

    # Gather all chunks and trace their mapping to unique_texts
    all_chunks: List[str] = []
    text_slices: Dict[str, Tuple[int, int]] = {}

    for t in unique_texts:
        if sentence_level:
            chunks = _split_sentences(t)
            if not chunks:
                chunks = [t]
        else:
            chunks = [t]

        start_idx = len(all_chunks)
        all_chunks.extend(chunks)
        end_idx = len(all_chunks)
        text_slices[t] = (start_idx, end_idx)

    # Batch encode all chunks together
    if all_chunks:
        all_embs = enc.encode(all_chunks, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    else:
        all_embs = np.empty((0, 384))

    # Evaluate aspects for each text using slice of embeddings
    unique_results = {}
    for t in unique_texts:
        start_idx, end_idx = text_slices[t]
        chunk_embs = all_embs[start_idx:end_idx]

        aspect_results: Dict[str, Dict[str, Any]] = {}

        for aspect, poles in embeddings.items():
            pos_scores_list: List[float] = []
            neg_scores_list: List[float] = []
            net_scores_list: List[float] = []
            rel_scores_list: List[float] = []

            for chunk_emb in chunk_embs:
                pos_sim, neg_sim, net = _signed_aspect_score(
                    chunk_emb, poles["positive"], poles["negative"]
                )
                rel = max(pos_sim, neg_sim)
                pos_scores_list.append(pos_sim)
                neg_scores_list.append(neg_sim)
                net_scores_list.append(net)
                rel_scores_list.append(rel)

            weights = np.array(rel_scores_list, dtype=float)
            w_sum = weights.sum()
            if w_sum > 0:
                weighted_net = float(np.dot(net_scores_list, weights) / w_sum)
            else:
                weighted_net = float(np.mean(net_scores_list)) if net_scores_list else 5.0

            relevance = float(np.max(rel_scores_list)) if rel_scores_list else 0.0
            detected  = relevance > 0.28

            aspect_results[aspect] = {
                "score_10":     round(weighted_net, 4),
                "positive_sim": round(float(np.mean(pos_scores_list)), 4) if pos_scores_list else 0.0,
                "negative_sim": round(float(np.mean(neg_scores_list)), 4) if neg_scores_list else 0.0,
                "relevance":    round(relevance, 4),
                "detected":     detected,
            }

        unique_results[t] = aspect_results

    return [{"text": t, "aspects": unique_results[t]} for t in texts]


# ── Dashboard backward-compat helpers ────────────────────────────────────────
def detect_aspects(
    text: str,
    model_name: str = DEFAULT_ENCODER,
    top_k: int = 3,
    threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """Generic top-k aspect detection (used by the Streamlit dashboard)."""
    enc = _load_encoder(model_name)
    if enc is None:
        return _detect_aspects_rule_based(text, top_k, threshold)
    compat_embs = _build_compat_embeddings(model_name)
    if compat_embs is None:
        return _detect_aspects_rule_based(text, top_k, threshold)
    text_emb = enc.encode(text, convert_to_numpy=True)

    scores: Dict[str, float] = {}
    for asp, asp_emb in compat_embs.items():
        sim = _cosine_sim(text_emb, asp_emb)
        if sim >= threshold:
            scores[asp] = round(sim, 4)

    sorted_asp = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"aspect": a, "score": s} for a, s in sorted_asp[:top_k]]


def analyze_aspects_batch(
    texts: List[str],
    model_name: str = DEFAULT_ENCODER,
    top_k: int = 3,
    threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """Batch version of detect_aspects (dashboard compatibility)."""
    return [
        {"text": t, "aspects": detect_aspects(t, model_name, top_k, threshold)}
        for t in texts
    ]


def aspect_frequency_table(batch_results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count aspect mention frequency across a batch (dashboard compatibility)."""
    freq: Dict[str, int] = {a: 0 for a in _COMPAT_ASPECTS}
    for item in batch_results:
        for asp_info in item.get("aspects", []):
            asp = asp_info["aspect"]
            freq[asp] = freq.get(asp, 0) + 1
    return freq


def _score_aspects_rule_based(text: str) -> Dict[str, Dict[str, Any]]:
    """Lightweight rule-based aspect scorer fallback for low-memory environments."""
    lower = text.lower()
    
    aspect_keywords = {
        "Communication": {
            "pos": ["clear", "clarity", "explain", "structure", "organized", "pace", "simple", "analogy", "analogies", "articulate"],
            "neg": ["fast", "unclear", "vague", "mumble", "confusing", "confusion", "jargon", "speed", "rapid", "mumbling"]
        },
        "Subject Knowledge": {
            "pos": ["knowledge", "expertise", "expert", "research", "industry", "theory", "competent", "factual", "academic", "deep"],
            "neg": ["outdated", "textbook", "error", "errors", "mistake", "mistakes", "unprepared", "inaccurate", "wrong"]
        },
        "Engagement": {
            "pos": ["interactive", "fun", "engage", "engaging", "discussion", "quiz", "enthusiasm", "enthusiastic", "passionate", "participation", "participate", "motivate", "motivation"],
            "neg": ["boring", "monotonous", "sleep", "dry", "one-sided", "passive", "uninteresting", "lecture-only"]
        },
        "Responsiveness": {
            "pos": ["responsive", "available", "office hours", "email", "reply", "helpful", "supportive", "approachable", "query", "queries", "support"],
            "neg": ["unresponsive", "late", "delayed", "ignore", "ignored", "dismissive", "unapproachable", "unavailable"]
        },
        "Assignment Quality": {
            "pos": ["assignment", "assignments", "grading", "grade", "grades", "rubric", "feedback", "practical", "real-world", "exam", "exams"],
            "neg": ["excessive", "workload", "inconsistent", "delayed grades", "no feedback", "hard grading"]
        }
    }
    
    results: Dict[str, Dict[str, Any]] = {}
    for aspect, kw in aspect_keywords.items():
        pos_count = sum(1 for w in kw["pos"] if w in lower)
        neg_count = sum(1 for w in kw["neg"] if w in lower)
        
        relevance = min(1.0, (pos_count + neg_count) / 3.0)
        detected = relevance > 0.15
        
        if pos_count == 0 and neg_count == 0:
            score = 5.0
            pos_sim = 0.0
            neg_sim = 0.0
        else:
            diff = pos_count - neg_count
            score = 5.0 + (diff * 2.0)
            score = max(1.0, min(9.0, score))
            pos_sim = min(1.0, pos_count / 3.0)
            neg_sim = min(1.0, neg_count / 3.0)
            
        results[aspect] = {
            "score_10": round(score, 2),
            "positive_sim": round(pos_sim, 2),
            "negative_sim": round(neg_sim, 2),
            "relevance": round(relevance, 2),
            "detected": detected
        }
        
    return results


def _detect_aspects_rule_based(text: str, top_k: int = 3, threshold: float = 0.25) -> List[Dict[str, Any]]:
    """Lightweight rule-based aspect detector fallback for low-memory environments."""
    lower = text.lower()
    aspect_keywords = {
        "Communication": ["clear", "clarity", "explain", "structure", "pace", "mumble", "confusing", "vague", "analogies", "articulate"],
        "Subject Knowledge": ["knowledge", "expertise", "expert", "research", "theory", "outdated", "textbook", "error", "errors", "mistake"],
        "Engagement": ["interactive", "fun", "engage", "discussion", "quiz", "enthusiasm", "passionate", "participation", "boring", "monotonous"],
        "Responsiveness": ["responsive", "available", "office hours", "email", "reply", "helpful", "unresponsive", "late", "delayed"],
        "Assignment Quality": ["assignment", "assignments", "grading", "grade", "rubric", "workload", "exams", "inconsistent"]
    }
    
    scores: Dict[str, float] = {}
    for aspect, keywords in aspect_keywords.items():
        matches = sum(1 for w in keywords if w in lower)
        if matches > 0:
            sim = min(0.95, 0.3 + (matches * 0.15))
            if sim >= threshold:
                scores[aspect] = round(sim, 4)
                
    sorted_asp = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"aspect": a, "score": s} for a, s in sorted_asp[:top_k]]
