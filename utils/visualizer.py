"""
utils/visualizer.py
-------------------
Plotly chart factory for the Teacher Feedback Analytics Dashboard.

All charts follow a consistent dark-theme design system with:
  - Background: #0f1117
  - Card surface: #1a1d2e
  - Accent gradient: #6c63ff → #f093fb
  - Font: Inter / system sans-serif
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d2e",
    "grid": "#2a2d3e",
    "text": "#e2e8f0",
    "subtext": "#94a3b8",
    "positive": "#4ade80",
    "neutral": "#facc15",
    "negative": "#f87171",
    "accent1": "#6c63ff",
    "accent2": "#f093fb",
    "accent3": "#4facfe",
    "accent4": "#43e97b",
}

SENTIMENT_COLORS = {
    "Positive": COLORS["positive"],
    "Neutral": COLORS["neutral"],
    "Negative": COLORS["negative"],
}

CHART_LAYOUT = dict(
    paper_bgcolor=COLORS["card"],
    plot_bgcolor=COLORS["card"],
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"], size=13),
    margin=dict(l=16, r=16, t=48, b=16),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=COLORS["grid"],
        font=dict(color=COLORS["subtext"]),
    ),
    xaxis=dict(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
        tickfont=dict(color=COLORS["subtext"]),
    ),
    yaxis=dict(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
        tickfont=dict(color=COLORS["subtext"]),
    ),
)


def _apply_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply the common dark-theme layout to any figure."""
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text=title,
            font=dict(size=15, color=COLORS["text"]),
            x=0.02,
        ),
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# 1. Sentiment Donut Chart
# ---------------------------------------------------------------------------
def sentiment_donut(
    counts: Dict[str, int],
    title: str = "Sentiment Distribution",
) -> go.Figure:
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SENTIMENT_COLORS.get(l, COLORS["accent1"]) for l in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.65,
            marker=dict(colors=colors, line=dict(color=COLORS["card"], width=3)),
            textinfo="label+percent",
            textfont=dict(size=13, color=COLORS["text"]),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )
    )
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:11px'>responses</span>",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=18, color=COLORS["text"]),
        align="center",
    )
    return _apply_layout(fig, title, height=360)


# ---------------------------------------------------------------------------
# 2. Aspect Radar Chart
# ---------------------------------------------------------------------------
def aspect_radar(
    aspect_scores: Dict[str, float],
    teacher_name: str = "Teacher",
) -> go.Figure:
    if not aspect_scores:
        return go.Figure()

    categories = list(aspect_scores.keys())
    values = list(aspect_scores.values())
    # Close the radar loop
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor=f"rgba(108, 99, 255, 0.25)",
            line=dict(color=COLORS["accent1"], width=2),
            marker=dict(color=COLORS["accent2"], size=7),
            name=teacher_name,
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
        polar=dict(
            bgcolor=COLORS["card"],
            gridshape="circular",
            radialaxis=dict(
                range=[0, 100],
                tickfont=dict(color=COLORS["subtext"], size=10),
                gridcolor=COLORS["grid"],
                linecolor=COLORS["grid"],
            ),
            angularaxis=dict(
                tickfont=dict(color=COLORS["text"], size=11),
                gridcolor=COLORS["grid"],
                linecolor=COLORS["grid"],
            ),
        ),
        title=dict(
            text=f"Aspect Performance — {teacher_name}",
            font=dict(size=15, color=COLORS["text"]),
            x=0.02,
        ),
        height=420,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Teacher Ranking Bar Chart
# ---------------------------------------------------------------------------
def teacher_ranking_bar(ranked_teachers: List[Dict[str, Any]]) -> go.Figure:
    if not ranked_teachers:
        return go.Figure()

    names = [t["teacher"] for t in ranked_teachers]
    scores = [t["score"] for t in ranked_teachers]
    grades = [t.get("grade", "N/A") for t in ranked_teachers]

    # Colour gradient from low to high
    norm_scores = np.array(scores)
    colors = px.colors.sample_colorscale(
        "Viridis", [s / 100 for s in norm_scores]
    )

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale="Viridis",
                cmin=0,
                cmax=100,
                colorbar=dict(
                    title=dict(text="Score", font=dict(color=COLORS["subtext"])),
                    tickfont=dict(color=COLORS["subtext"]),
                ),
            ),
            text=[f"{s:.1f} ({g})" for s, g in zip(scores, grades)],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=12),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
        )
    )
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        xaxis=dict(range=[0, 110], title="Performance Score"),
        yaxis=dict(autorange="reversed"),
        title=dict(
            text="Teacher Performance Ranking",
            font=dict(size=15, color=COLORS["text"]),
            x=0.02,
        ),
        height=max(300, len(names) * 55 + 80),
        margin=dict(l=16, r=90, t=48, b=16),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Trend Line Chart
# ---------------------------------------------------------------------------
def trend_line(
    trend_df: pd.DataFrame,
    title: str = "Sentiment Score Trend Over Time",
) -> go.Figure:
    if trend_df.empty:
        return go.Figure()

    fig = go.Figure()

    # Area fill
    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["mean_score"],
            mode="lines+markers",
            name="Avg Score",
            line=dict(color=COLORS["accent1"], width=2.5),
            marker=dict(color=COLORS["accent2"], size=7, line=dict(color=COLORS["card"], width=2)),
            fill="tozeroy",
            fillcolor="rgba(108, 99, 255, 0.15)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Score: %{y:.1f}<extra></extra>",
        )
    )

    if "count" in trend_df.columns:
        fig.add_trace(
            go.Bar(
                x=trend_df["date"],
                y=trend_df["count"],
                name="# Responses",
                yaxis="y2",
                marker=dict(color="rgba(240, 147, 251, 0.25)"),
                hovertemplate="Responses: %{y}<extra></extra>",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title="Responses",
                overlaying="y",
                side="right",
                showgrid=False,
                tickfont=dict(color=COLORS["subtext"]),
            )
        )

    return _apply_layout(fig, title, height=380)


# ---------------------------------------------------------------------------
# 5. Aspect Frequency Horizontal Bar
# ---------------------------------------------------------------------------
def aspect_frequency_bar(freq: Dict[str, int], title: str = "Aspect Mentions") -> go.Figure:
    if not freq:
        return go.Figure()

    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    aspects = [i[0] for i in sorted_items]
    counts = [i[1] for i in sorted_items]
    accent_palette = [
        COLORS["accent1"], COLORS["accent2"], COLORS["accent3"], COLORS["accent4"],
        COLORS["positive"], COLORS["neutral"], COLORS["negative"], COLORS["accent1"],
    ]
    bar_colors = [accent_palette[i % len(accent_palette)] for i in range(len(aspects))]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=aspects,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color=COLORS["card"], width=1)),
            text=counts,
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
            hovertemplate="%{y}: %{x} mentions<extra></extra>",
        )
    )
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        title=dict(text=title, font=dict(size=15, color=COLORS["text"]), x=0.02),
        height=max(300, len(aspects) * 45 + 80),
        margin=dict(l=16, r=60, t=48, b=16),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Score Distribution Histogram
# ---------------------------------------------------------------------------
def score_histogram(scores: List[float], title: str = "Score Distribution") -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=scores,
            nbinsx=20,
            marker=dict(
                color=scores,
                colorscale="Plasma",
                line=dict(color=COLORS["card"], width=1),
            ),
            hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    return _apply_layout(fig, title, height=340)


# ---------------------------------------------------------------------------
# 7. Sentiment Stacked Bar by Teacher
# ---------------------------------------------------------------------------
def sentiment_stacked_bar(
    df: pd.DataFrame,
    teacher_col: str = "teacher_name",
    sentiment_col: str = "sentiment_label",
) -> go.Figure:
    if df.empty or teacher_col not in df.columns or sentiment_col not in df.columns:
        return go.Figure()

    grouped = (
        df.groupby([teacher_col, sentiment_col])
        .size()
        .reset_index(name="count")
    )
    pivot = grouped.pivot(index=teacher_col, columns=sentiment_col, values="count").fillna(0)

    fig = go.Figure()
    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in pivot.columns:
            fig.add_trace(
                go.Bar(
                    name=sentiment,
                    x=pivot.index.tolist(),
                    y=pivot[sentiment].tolist(),
                    marker_color=SENTIMENT_COLORS[sentiment],
                    hovertemplate=f"<b>%{{x}}</b><br>{sentiment}: %{{y}}<extra></extra>",
                )
            )
    fig.update_layout(barmode="stack")
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(
            text="Sentiment Breakdown by Teacher",
            font=dict(size=15, color=COLORS["text"]),
            x=0.02,
        ),
        height=400,
        xaxis=dict(tickangle=-25),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. KPI Gauge
# ---------------------------------------------------------------------------
def score_gauge(score: float, title: str = "Overall Score") -> go.Figure:
    color = (
        COLORS["positive"] if score >= 70
        else COLORS["neutral"] if score >= 50
        else COLORS["negative"]
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 70, "valueformat": ".1f"},
            number={"font": {"size": 40, "color": color}, "suffix": ""},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": COLORS["subtext"],
                    "tickfont": {"color": COLORS["subtext"]},
                },
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": COLORS["grid"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(248,113,113,0.15)"},
                    {"range": [50, 70], "color": "rgba(250,204,21,0.15)"},
                    {"range": [70, 100], "color": "rgba(74,222,128,0.15)"},
                ],
                "threshold": {
                    "line": {"color": COLORS["accent2"], "width": 3},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
            title={"text": title, "font": {"size": 14, "color": COLORS["subtext"]}},
        )
    )
    fig.update_layout(
        paper_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Feedback Volume Bar Chart
# ---------------------------------------------------------------------------
def feedback_volume_bar(
    df: pd.DataFrame,
    title: str = "Feedback Volume by Teacher",
) -> go.Figure:
    if df.empty or "teacher_name" not in df.columns:
        return go.Figure()

    counts = df["teacher_name"].value_counts().reset_index()
    counts.columns = ["Teacher", "Feedback Count"]

    fig = go.Figure(
        go.Bar(
            x=counts["Feedback Count"],
            y=counts["Teacher"],
            orientation="h",
            marker=dict(
                color=counts["Feedback Count"],
                colorscale="Sunset",
                cmin=0,
                colorbar=dict(
                    title=dict(text="Volume", font=dict(color=COLORS["subtext"])),
                    tickfont=dict(color=COLORS["subtext"]),
                ),
            ),
            text=counts["Feedback Count"],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=12),
            hovertemplate="<b>%{y}</b><br>Reviews: %{x}<extra></extra>",
        )
    )
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        xaxis=dict(title="Number of Feedback Entries"),
        yaxis=dict(autorange="reversed"),
        title=dict(
            text=title,
            font=dict(size=15, color=COLORS["text"]),
            x=0.02,
        ),
        height=max(300, len(counts) * 45 + 80),
        margin=dict(l=16, r=60, t=48, b=16),
    )
    return fig
