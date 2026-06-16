---
title: Teacher Feedback Analytics
emoji: 🎓
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Teacher Feedback Analytics Dashboard

AI-powered dashboard for analysing student feedback about teachers using HuggingFace Transformers and Sentence Transformers.

---

## Project Structure

```
teacher-feedback-analytics/
├── app.py                      # Streamlit dashboard entry point
├── demo_pipeline.py            # Standalone NLP pipeline demo
├── requirements.txt
├── data/
│   ├── feedback_dataset.csv    # 500-entry sample dataset
│   └── generate_dataset.py     # Dataset generator script
├── nlp/
│   ├── __init__.py             # Package exports: analyze, analyze_batch
│   ├── pipeline.py             # ★ Main orchestrator — use this
│   ├── sentiment.py            # Sentiment analysis (RoBERTa 3-class)
│   ├── aspects.py              # ABSA — 5 aspect scoring (sentence-transformers)
│   ├── scoring.py              # Composite scoring engine
│   └── summarizer.py          # Abstractive summarization + keyword extraction
├── utils/
│   ├── data_loader.py          # CSV / Excel ingestion + sample data
│   ├── visualizer.py           # Plotly chart factory (dark theme)
│   └── report_generator.py     # PDF / CSV export
├── tests/
│   └── test_pipeline.py        # 16-test suite
└── reports/                    # Auto-generated report outputs
```

---

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS / Linux

# 2. Install all dependencies
pip install -r requirements.txt
```

---

## Running the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## NLP Pipeline — Quick Start

### Single Feedback

```python
from nlp.pipeline import analyze

result = analyze(
    "The teacher explains concepts with exceptional clarity and responds "
    "to emails quickly. However, the assignment workload is excessive."
)

print(result)
```

**Output:**
```json
{
    "overall_score": 7.2,
    "communication_score": 9.1,
    "subject_knowledge_score": 7.8,
    "engagement_score": 7.3,
    "responsiveness_score": 8.4,
    "assignment_quality_score": 4.1,
    "strengths": ["clear explanations", "strong responsiveness"],
    "improvements": ["assignment workload"],
    "confidence": 0.87,
    "sentiment_label": "Positive",
    "is_mixed": true
}
```

### Batch Processing (with deduplication)

```python
from nlp.pipeline import analyze_batch

texts = [
    "Great teacher, very engaging and knowledgeable.",
    "Lectures are confusing and feedback is always delayed.",
    "Great teacher, very engaging and knowledgeable.",  # duplicate → reused
]

results = analyze_batch(texts)

for r in results:
    print(r["overall_score"], r["_is_duplicate"])
# 8.3  False
# 3.1  False
# 8.3  True   ← cached, not recomputed
```

### Custom Configuration

```python
from nlp.pipeline import analyze, PipelineConfig

cfg = PipelineConfig(
    sentiment_model="distilbert-base-uncased-finetuned-sst-2-english",  # faster
    encoder_model="all-MiniLM-L6-v2",
    batch_size=32,
    deduplicate=True,
    sentence_level=True,   # per-sentence ABSA (more accurate on mixed feedback)
)

result = analyze("Your feedback text here.", config=cfg)
```

---

## Output Schema

| Field | Type | Range | Description |
|---|---|---|---|
| `overall_score` | float | 0–10 | Weighted composite score |
| `communication_score` | float | 0–10 | Clarity, pace, articulation |
| `subject_knowledge_score` | float | 0–10 | Depth, accuracy, currency |
| `engagement_score` | float | 0–10 | Interactivity, enthusiasm |
| `responsiveness_score` | float | 0–10 | Availability, email speed |
| `assignment_quality_score` | float | 0–10 | Design, workload, grading |
| `strengths` | list[str] | max 5 | Detected positive phrases |
| `improvements` | list[str] | max 5 | Detected areas to improve |
| `confidence` | float | 0–1 | Model confidence in sentiment |
| `sentiment_label` | str | — | `"Positive"` / `"Neutral"` / `"Negative"` |
| `is_mixed` | bool | — | True when sentiment is ambiguous |
| `_text` | str | — | Original input text |
| `_sentiment_probs` | dict | — | {Positive, Neutral, Negative} → float |
| `_aspect_detail` | dict | — | Per-aspect score, relevance, detected |
| `_is_duplicate` | bool | — | True for repeat texts in batch |

---

## Running Tests

```bash
# With pytest
pytest tests/test_pipeline.py -v

# Plain Python (no pytest needed)
python tests/test_pipeline.py
```

**16 tests** covering: schema validation, score ranges, deduplication, mixed sentiment detection, JSON serializability, empty inputs, and all module imports.

---

## Running the Pipeline Demo

```bash
# Interactive demo with 4 representative feedback texts
python demo_pipeline.py

# Analyse a custom text
python demo_pipeline.py "Your custom feedback text here."

# CLI mode via pipeline.py directly
python -m nlp.pipeline "The teacher is excellent but gives too much homework."
```

---

## Models Used

| Model | Purpose | Size |
|---|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment classification (3-class) | ~500 MB |
| `all-MiniLM-L6-v2` | Sentence embeddings for ABSA | ~80 MB |
| `facebook/bart-large-cnn` | Abstractive summarization | ~1.6 GB |
| `sshleifer/distilbart-cnn-12-6` | Summarization fallback (faster) | ~300 MB |

> Models are downloaded automatically on first use and cached by HuggingFace.

---

## Architecture

```
analyze(text)
    │
    ├─▶ sentiment.analyze_sentiment()   → polarity score, probs, is_mixed
    │       Model: RoBERTa-3class
    │
    ├─▶ aspects.score_aspects()         → per-aspect signed similarity scores
    │       Model: all-MiniLM-L6-v2
    │       Method: top-3 mean cosine sim vs positive/negative anchor sentences
    │
    ├─▶ scoring.score_single()          → canonical JSON with blended scores
    │       Blend: alpha * polarity + (1-alpha) * ABSA (relevance-weighted)
    │
    └─▶ summarizer.extract_strengths/improvements()
            Method: keyword matching against curated phrase maps
```

---

## Dataset

The `data/feedback_dataset.csv` contains **500 unique entries** across:
- **22 teachers** across 22 subjects
- **4 semesters** (Spring/Fall 2023–2024)
- **5 feedback dimensions**: Communication, Engagement, Responsiveness, Assignment Quality, Subject Knowledge
- Mixed sentiment ratio: ~45% positive, ~25% negative, ~30% neutral/mixed
