"""
utils package — Teacher Feedback Analytics

Quick start
-----------
    # Full pipeline (raw CSV → aggregated scores)
    from utils.aggregator import aggregate
    import pandas as pd

    df  = pd.read_csv("data/feedback_dataset.csv")
    agg = aggregate(df)          # one row per teacher
    print(agg[["teacher_name", "overall_score", "grade", "total_reviews"]])

    # Pre-scored DataFrame (skip pipeline rerun)
    from utils.aggregator import aggregate_from_scores
    agg = aggregate_from_scores(scored_df)

    # Cohort-level statistics
    from utils.aggregator import cohort_summary
    print(cohort_summary(agg))

    # Reshape to long format for charts
    from utils.aggregator import pivot_aspect_scores
    long_df = pivot_aspect_scores(agg)
"""

from utils.aggregator import (
    aggregate,
    aggregate_from_scores,
    run_pipeline_on_df,
    pivot_aspect_scores,
    teacher_report_dict,
    cohort_summary,
    AggregationConfig,
)

from utils.data_loader import (
    load_csv,
    load_excel,
    generate_sample_dataset,
    validate_dataset,
)

__all__ = [
    # aggregator
    "aggregate",
    "aggregate_from_scores",
    "run_pipeline_on_df",
    "pivot_aspect_scores",
    "teacher_report_dict",
    "cohort_summary",
    "AggregationConfig",
    # data loader
    "load_csv",
    "load_excel",
    "generate_sample_dataset",
    "validate_dataset",
]
