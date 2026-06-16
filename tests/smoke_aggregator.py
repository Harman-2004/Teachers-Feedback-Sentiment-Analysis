"""
Smoke test: full CSV -> pre-scored DataFrame -> aggregate -> report
"""
import sys, json
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

from utils.aggregator import (
    aggregate_from_scores, pivot_aspect_scores,
    teacher_report_dict, cohort_summary, AggregationConfig
)

# ── Load real dataset, inject mock scores ────────────────────────────────────
df = pd.read_csv('data/feedback_dataset.csv')
n_teachers = df['teacher_name'].nunique()
n = len(df)
print(f'Loaded {n} rows, {n_teachers} teachers')

rng = np.random.default_rng(42)

df['communication_score']      = rng.uniform(3.0, 9.5, n).round(2)
df['subject_knowledge_score']  = rng.uniform(3.0, 9.5, n).round(2)
df['engagement_score']         = rng.uniform(3.0, 9.0, n).round(2)
df['responsiveness_score']     = rng.uniform(3.0, 9.0, n).round(2)
df['assignment_quality_score'] = rng.uniform(2.0, 8.5, n).round(2)
df['confidence']               = rng.uniform(0.55, 0.95, n).round(3)
df['sentiment_label']          = rng.choice(
    ['Positive', 'Neutral', 'Negative'], n, p=[0.5, 0.3, 0.2]
)
df['is_mixed']          = rng.random(n) < 0.15
df['strengths_list']    = [['clear explanations'] if rng.random() > 0.5 else [] for _ in range(n)]
df['improvements_list'] = [['assignment workload'] if rng.random() > 0.6 else [] for _ in range(n)]

# ── Aggregate ────────────────────────────────────────────────────────────────
cfg = AggregationConfig()   # 25/25/20/15/15 weights
agg = aggregate_from_scores(df, config=cfg)

print()
print('=== Aggregated Teacher Scores (one row per teacher) ===')
display_cols = [
    'rank', 'teacher_name', 'total_reviews', 'overall_score',
    'communication_score', 'subject_knowledge_score',
    'engagement_score', 'responsiveness_score',
    'assignment_quality_score', 'confidence_score',
    'confidence_band', 'grade'
]
print(agg[display_cols].to_string(index=False))

print()
print('=== Cohort Summary ===')
summary = cohort_summary(agg)
for k, v in summary.items():
    print(f'  {k:<28}: {v}')

print()
print('=== Pivot (long format — first 10 rows) ===')
long = pivot_aspect_scores(agg)
print(long[['teacher_name', 'aspect', 'score']].head(10).to_string(index=False))

print()
print('=== Single Teacher Report Dict (rank 1) ===')
top_teacher = agg.iloc[0]['teacher_name']
report = teacher_report_dict(agg, top_teacher)
public_keys = [
    'teacher_name', 'overall_score', 'grade', 'descriptor',
    'total_reviews', 'confidence_score', 'confidence_band',
    'positive_pct', 'negative_pct', 'strengths', 'improvements'
]
for k in public_keys:
    print(f'  {k:<28}: {report[k]}')

print()
print('=== Weight Validation ===')
weight_sum = sum(cfg.weights.values())
print(f'  Weight sum        : {weight_sum:.4f}  (should be 1.0000)')
for asp, w in cfg.weights.items():
    print(f'  {asp:<20}: {w*100:.0f}%')

print()
print('=== Shape Checks ===')
print(f'  agg rows          : {len(agg)}  (expected: {n_teachers})')
print(f'  long rows         : {len(long)}  (expected: {n_teachers * 5})')
print(f'  max rank          : {agg["rank"].max()} == n_teachers: {agg["rank"].max() == len(agg)}')

score_ok = all(
    (agg[c].between(0, 10)).all()
    for c in ['overall_score', 'communication_score', 'engagement_score',
              'responsiveness_score', 'assignment_quality_score']
)
print(f'  all scores in [0,10]: {score_ok}')
conf_ok = agg['confidence_score'].between(0, 1).all()
print(f'  confidence in [0,1] : {conf_ok}')

print()
print('DONE — full smoke test complete.')
