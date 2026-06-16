"""
app.py
------
Teacher Feedback Analytics Dashboard
=====================================
A Streamlit application for AI-powered analysis of student feedback
about teachers using sentiment analysis, aspect detection, and
abstractive summarization.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup (so local packages are importable)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Teacher Feedback Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Teacher Feedback Analytics Dashboard — powered by AI",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark + light theme
# ---------------------------------------------------------------------------
_DARK_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ══ DARK THEME ══ */
:root {
  --bg:        #0f1117;
  --card:      #1a1d2e;
  --card2:     #21253a;
  --border:    #2a2d3e;
  --text:      #e2e8f0;
  --subtext:   #94a3b8;
  --accent1:   #6c63ff;
  --accent2:   #f093fb;
  --positive:  #4ade80;
  --neutral:   #facc15;
  --negative:  #f87171;
}
[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,.stApp {
  background-color: #0f1117 !important;
}
[data-testid="stHeader"] { background-color: #0f1117 !important; }
[data-testid="stSidebar"],[data-testid="stSidebarContent"] {
  background: linear-gradient(180deg, #12162a 0%, #0f1117 100%) !important;
}"""

_LIGHT_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ══ LIGHT THEME — Lavender & White Premium ══ */
:root {
  --bg:        #f5f3ff;
  --card:      #ffffff;
  --card2:     #faf9ff;
  --border:    #e4e0fb;
  --text:      #1e1b4b;
  --subtext:   #5b5ea6;
  --accent1:   #7c3aed;
  --accent2:   #db2777;
  --positive:  #059669;
  --neutral:   #d97706;
  --negative:  #e11d48;
}

/* ── Override EVERY Streamlit container ── */
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stVerticalBlockBorderWrapper"],
.main, .stApp, section.main, div.main,
.block-container {
  background-color: #f5f3ff !important;
  color: #1e1b4b !important;
}
[data-testid="stHeader"] {
  background-color: #f5f3ff !important;
  border-bottom: 1px solid #e4e0fb !important;
}

/* ── Light sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
  background: linear-gradient(160deg, #ede9fe 0%, #f5f3ff 100%) !important;
  border-right: 1px solid #e4e0fb !important;
}

/* ── Force all text to dark in light mode ── */
[data-testid="stApp"] p,
[data-testid="stApp"] span:not(.badge-positive):not(.badge-negative):not(.badge-neutral),
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stTextInput label, .stSlider label, .stToggle label,
.stRadio label, .stMultiSelect label, .stSelectbox label,
.stFileUploader label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div:not(.theme-chip):not(.stButton) {
  color: #1e1b4b !important;
}

/* ── Widget input backgrounds ── */
.stTextInput > div > div,
.stTextArea > div > div,
.stSelectbox > div > div > div,
.stMultiSelect > div > div {
  background-color: #ffffff !important;
  border-color: #e4e0fb !important;
  color: #1e1b4b !important;
}

/* ── Dataframe in light ── */
.stDataFrame, [data-testid="stDataFrame"] { background: #ffffff !important; }

/* ── Tab list in light ── */
.stTabs [data-baseweb="tab-list"] { background: #ede9fe !important; }"""

_SHARED_CSS = """
/* ── Global ── */
html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}
.main .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

/* ── Sidebar (shared fallback — theme-specific overrides take priority) ── */
[data-testid="stSidebar"] {
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: var(--accent2) !important; }

/* ── Theme toggle chip ── */
.theme-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.85rem;
  border-radius: 99px;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  cursor: default;
  background: linear-gradient(135deg, var(--accent1), var(--accent2));
  color: #fff;
  margin-bottom: 0.5rem;
}

/* ── Header ── */
.dashboard-header {
  background: linear-gradient(135deg, var(--card) 0%, var(--card2) 50%, var(--card) 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem 2.5rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  gap: 2rem;
}
.dashboard-header::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -20%;
  width: 60%;
  height: 200%;
  background: radial-gradient(ellipse, rgba(108,99,255,0.15) 0%, transparent 70%);
  pointer-events: none;
}
.dashboard-header-text { flex: 1; }
.dashboard-header h1 {
  font-size: 2.2rem;
  font-weight: 800;
  background: linear-gradient(135deg, #a78bfa, #f093fb);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.3rem 0;
}
.dashboard-header p { color: var(--subtext); margin: 0; font-size: 1rem; }
.dashboard-hero-img {
  flex-shrink: 0;
  width: 210px;
  height: 140px;
  object-fit: contain;
  border-radius: 12px;
  opacity: 0.92;
  filter: drop-shadow(0 4px 24px rgba(108,99,255,0.4));
  animation: float 4s ease-in-out infinite;
}
@keyframes float {
  0%,100% { transform: translateY(0px);  }
  50%      { transform: translateY(-8px); }
}

/* ── KPI Cards ── */
.kpi-card {
  background: linear-gradient(135deg, var(--card) 0%, var(--card2) 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.25rem 1.5rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: var(--accent1); }
.kpi-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--accent1), var(--accent2));
}
.kpi-value {
  font-size: 2.4rem;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 0.25rem;
}
.kpi-label {
  font-size: 0.8rem;
  color: var(--subtext);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 500;
}
.kpi-sub {
  font-size: 0.9rem;
  color: var(--subtext);
  margin-top: 0.2rem;
}

/* ── Section headers ── */
.section-header {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text);
  margin: 1.5rem 0 0.75rem 0;
  padding-bottom: 0.4rem;
  border-bottom: 2px solid var(--border);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.section-header span { color: var(--accent1); }

/* ── Feedback cards ── */
.feedback-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  margin-bottom: 0.75rem;
  border-left: 4px solid var(--accent1);
  transition: border-color 0.2s;
}
.feedback-card.positive { border-left-color: var(--positive); }
.feedback-card.negative { border-left-color: var(--negative); }
.feedback-card.neutral  { border-left-color: var(--neutral); }
.feedback-text { color: var(--text); font-size: 0.92rem; line-height: 1.55; }
.feedback-meta { color: var(--subtext); font-size: 0.78rem; margin-top: 0.4rem; }

/* ── Badge ── */
.badge {
  display: inline-block;
  padding: 0.2rem 0.65rem;
  border-radius: 99px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.badge-positive { background: rgba(74,222,128,0.15); color: var(--positive); }
.badge-negative { background: rgba(248,113,113,0.15); color: var(--negative); }
.badge-neutral  { background: rgba(250,204,21,0.15);  color: var(--neutral);  }

/* ── Summary box ── */
.summary-box {
  background: linear-gradient(135deg, rgba(108,99,255,0.1), rgba(240,147,251,0.08));
  border: 1px solid rgba(108,99,255,0.35);
  border-radius: 14px;
  padding: 1.25rem 1.5rem;
  color: var(--text);
  line-height: 1.7;
  font-size: 0.95rem;
}

/* ── Progress bar ── */
.aspect-bar-wrap { margin: 0.4rem 0; }
.aspect-label { font-size: 0.85rem; color: var(--subtext); margin-bottom: 0.15rem; }
.aspect-bar-bg {
  background: var(--border);
  border-radius: 99px;
  height: 8px;
  overflow: hidden;
}
.aspect-bar-fill {
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, var(--accent1), var(--accent2));
  transition: width 0.6s ease;
}

/* ── Table ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px; }
thead tr th { background: var(--card2) !important; color: var(--text) !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  padding: 0.5rem 1.5rem !important;
  transition: opacity 0.2s, transform 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── Selectbox / Input ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--card2) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--card); border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
  color: var(--subtext) !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
  color: white !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent1) !important; }

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""


def _inject_theme(dark_mode: bool):
    """Inject dark or light theme variables plus all shared styles."""
    base = _DARK_CSS if dark_mode else _LIGHT_CSS
    st.markdown(base + _SHARED_CSS, unsafe_allow_html=True)


# Theme will be injected after sidebar reads user preference (see main())


# ---------------------------------------------------------------------------
# Lazy imports with error handling
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_nlp_modules():
    """Load NLP modules once and cache them."""
    try:
        from nlp.sentiment import analyze_sentiment, sentiment_summary
        from nlp.aspects import analyze_aspects_batch, aspect_frequency_table
        from nlp.scoring import compute_teacher_score, rank_teachers, trend_analysis
        from nlp.summarizer import summarize_feedback, summarize_by_sentiment, extract_keywords, summarize_teacher_feedback
        return {
            "analyze_sentiment": analyze_sentiment,
            "sentiment_summary": sentiment_summary,
            "analyze_aspects_batch": analyze_aspects_batch,
            "aspect_frequency_table": aspect_frequency_table,
            "compute_teacher_score": compute_teacher_score,
            "rank_teachers": rank_teachers,
            "trend_analysis": trend_analysis,
            "summarize_feedback": summarize_feedback,
            "summarize_by_sentiment": summarize_by_sentiment,
            "extract_keywords": extract_keywords,
            "summarize_teacher_feedback": summarize_teacher_feedback,
        }
    except (ImportError, OSError, Exception) as e:
        st.error(f"NLP module import error: {e}")
        return None


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "df": None,
        "analysis_done": False,
        "sentiment_results": [],
        "aspect_results": [],
        "teacher_scores": {},
        "ranked_teachers": [],
        "selected_teacher": None,
        "dataset_source": "sample",
        "teacher_summaries": {},
        "detailed_csv": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        # ── Branding ──
        st.markdown(
            """
            <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
              <div style='font-size:3rem'>🎓</div>
              <div style='font-size:1.1rem; font-weight:700; color:#a78bfa;'>Feedback Analytics</div>
              <div style='font-size:0.75rem; color:#64748b; margin-top:0.2rem;'>AI-Powered Dashboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Theme toggle ──
        st.markdown("---")
        st.markdown("### 🎨 Appearance")
        dark_mode = st.toggle(
            "🌙 Dark Mode",
            value=True,
            key="dark_mode_toggle",
            help="Switch between Dark and Light themes",
        )
        theme_label = "🌙 Dark Theme" if dark_mode else "☀️ Light Theme"
        st.markdown(
            f"<div class='theme-chip'>{theme_label}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### 📂 Data Source")

        source = st.radio(
            "Choose input method:",
            ["Use Sample Dataset", "Upload CSV / Excel", "Enter Feedback Manually"],
            key="data_source_radio",
            label_visibility="collapsed",
        )

        uploaded_file = None
        manual_text = None
        manual_teacher = None

        if source == "Upload CSV / Excel":
            uploaded_file = st.file_uploader(
                "Upload feedback file",
                type=["csv", "xlsx"],
                help="Required column: feedback_text (or similar). Optional: teacher_name, date, subject, rating",
            )
        elif source == "Enter Feedback Manually":
            manual_teacher = st.text_input("Teacher Name", value="Teacher A")
            manual_text = st.text_area(
                "Paste feedback (one per line):",
                height=180,
                placeholder="The teacher explains concepts very clearly...\nVery engaging sessions...",
            )

        st.markdown("---")
        st.markdown("### ⚙️ Analysis Options")

        run_summarizer = st.toggle("Enable AI Summarization", value=True,
                                   help="Uses BART model — slower but generates human-readable summaries")
        run_aspects = st.toggle("Enable Aspect Detection", value=True,
                                help="Uses sentence-transformers to identify pedagogical dimensions")
        use_fallback = st.toggle(
            "Fast Rule-Based Mode",
            value=True,
            help="Bypasses heavy deep learning models (PyTorch) to run analysis instantly. Recommended for low-resource environments or when encountering startup delays."
        )
        show_sample_size = st.slider(
            "Max feedback entries to analyse",
            min_value=20, max_value=500, value=150, step=10,
        )

        st.markdown("---")
        analyse_btn = st.button("🚀 Run Analysis", use_container_width=True)

        st.markdown("---")
        st.markdown(
            """
            <div style='color:#475569; font-size:0.75rem; text-align:center; line-height:1.6;'>
            <b style='color:#6c63ff;'>Models used</b><br>
            🤗 cardiffnlp/twitter-roberta<br>
            🤗 facebook/bart-large-cnn<br>
            🤗 all-MiniLM-L6-v2
            </div>
            """,
            unsafe_allow_html=True,
        )

    return source, uploaded_file, manual_text, manual_teacher, run_summarizer, run_aspects, use_fallback, show_sample_size, analyse_btn, dark_mode


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
def render_kpi_row(
    total: int,
    pos_pct: float,
    neg_pct: float,
    neu_pct: float,
    overall_score: float,
    grade: str,
):
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, str(total), "Total Responses", "#6c63ff", "📋"),
        (c2, f"{pos_pct:.1f}%", "Positive", "#4ade80", "😊"),
        (c3, f"{neu_pct:.1f}%", "Neutral", "#facc15", "😐"),
        (c4, f"{neg_pct:.1f}%", "Negative", "#f87171", "😔"),
        (c5, f"{overall_score:.1f}", f"Score  ·  {grade}", "#a78bfa", "🏆"),
    ]
    for col, value, label, color, icon in kpis:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div style="font-size:1.5rem">{icon}</div>
                  <div class="kpi-value" style="color:{color}">{value}</div>
                  <div class="kpi-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Feedback table with inline sentiment badges
# ---------------------------------------------------------------------------
def render_feedback_table(df: pd.DataFrame, max_rows: int = 50):
    st.markdown('<div class="section-header"><span>📝</span> Feedback Explorer</div>', unsafe_allow_html=True)

    cols_available = [c for c in ["teacher_name", "subject", "date", "rating", "feedback_text", "sentiment_label", "sentiment_score"] if c in df.columns]
    display_df = df[cols_available].head(max_rows).copy()

    if "sentiment_label" in display_df.columns:
        def label_html(lbl):
            cls = lbl.lower() if lbl else "neutral"
            return f'<span class="badge badge-{cls}">{lbl}</span>'
        display_df["sentiment_label"] = display_df["sentiment_label"].apply(label_html)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=320,
    )


def clear_detailed_csv():
    """Clear the cached detailed CSV report from session state when parameters change."""
    if "detailed_csv" in st.session_state:
        st.session_state["detailed_csv"] = None


def get_cached_teacher_summary(
    teacher_name: str,
    comments: List[str],
    aspect_scores: Dict[str, float],
    nlp_modules: Dict[str, Any],
    run_summarizer: bool,
) -> Dict[str, Any]:
    """
    Get or generate the AI summary for a teacher, using st.session_state for caching.
    """
    if "teacher_summaries" not in st.session_state:
        st.session_state["teacher_summaries"] = {}

    if not run_summarizer:
        return {"summary": "Summarization disabled.", "strengths": [], "improvements": []}

    import hashlib
    comments_hash = hashlib.sha256(" ".join(comments).encode("utf-8")).hexdigest()
    cache_key = f"{teacher_name}_{comments_hash}"

    if cache_key in st.session_state["teacher_summaries"]:
        return st.session_state["teacher_summaries"][cache_key]

    if "summarize_teacher_feedback" in nlp_modules:
        sum_res = nlp_modules["summarize_teacher_feedback"](comments, aspect_scores)
    else:
        sum_res = {"summary": "Summarization module not loaded.", "strengths": [], "improvements": []}

    st.session_state["teacher_summaries"][cache_key] = sum_res
    return sum_res


# ---------------------------------------------------------------------------
# Individual teacher view
# ---------------------------------------------------------------------------
def render_teacher_detail(
    teacher_name: str,
    nlp: dict,
    teacher_df: pd.DataFrame,
    run_summarizer: bool,
):
    from utils.visualizer import (
        aspect_radar, score_gauge, sentiment_donut, trend_line,
    )
    from utils.aggregator import aggregate_from_scores

    # Aggregate scores for this teacher
    agg_df = aggregate_from_scores(teacher_df)
    if agg_df.empty:
        st.warning(f"No aggregated data found for {teacher_name}.")
        return

    comments = teacher_df["feedback_text"].tolist()

    row = agg_df.iloc[0]

    # Extract metrics (scaled to 0-100 for gauge)
    overall_score = row["overall_score"] * 10
    grade = row["grade"]
    descriptor = row["descriptor"]

    # aspect scores (0-100)
    aspect_scores = {
        "Communication": row["communication_score"] * 10,
        "Subject Knowledge": row["subject_knowledge_score"] * 10,
        "Engagement": row["engagement_score"] * 10,
        "Responsiveness": row["responsiveness_score"] * 10,
        "Assignment Quality": row["assignment_quality_score"] * 10,
    }

    # sentiment counts
    counts = teacher_df["sentiment_label"].value_counts().to_dict()

    # Layout
    col_gauge, col_donut = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(
            score_gauge(overall_score, f"Overall Score"),
            use_container_width=True,
        )
        st.markdown(
            f"""<div style='text-align:center;'>
            <span style='font-size:1.8rem; font-weight:800; color:#a78bfa;'>{grade}</span>
            <span style='color:#94a3b8; margin-left:0.5rem;'>{descriptor}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_donut:
        st.plotly_chart(
            sentiment_donut(counts, "Sentiment Distribution"),
            use_container_width=True,
        )

    st.markdown("---")

    # Aspect scores radar + progress bars
    st.plotly_chart(
        aspect_radar(aspect_scores, teacher_name),
        use_container_width=True,
    )

    st.markdown('<div class="section-header"><span>📊</span> Aspect Scores</div>', unsafe_allow_html=True)
    for aspect, sc in sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True):
        st.markdown(
            f"""<div class="aspect-bar-wrap">
              <div class="aspect-label">{aspect} — {sc:.1f}/100</div>
              <div class="aspect-bar-bg"><div class="aspect-bar-fill" style="width:{min(sc,100)}%"></div></div>
            </div>""",
            unsafe_allow_html=True,
        )

    # AI Summary
    sum_res = {"summary": "Summarization disabled.", "strengths": [], "improvements": []}
    if run_summarizer:
        st.markdown('<div class="section-header"><span>🤖</span> AI Summary</div>', unsafe_allow_html=True)
        with st.spinner("Generating summary..."):
            sum_res = get_cached_teacher_summary(teacher_name, comments, aspect_scores, nlp, run_summarizer)

        st.markdown(f'<div class="summary-box">{sum_res["summary"]}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_str, col_imp = st.columns(2)
        with col_str:
            st.markdown("##### 🌟 Key Strengths")
            if sum_res["strengths"]:
                for s in sum_res["strengths"]:
                    st.markdown(f"- **{s.title()}**")
            else:
                st.markdown("*None identified.*")
        with col_imp:
            st.markdown("##### 🛠️ Areas for Improvement")
            if sum_res["improvements"]:
                for i in sum_res["improvements"]:
                    st.markdown(f"- **{i.capitalize()}**")
            else:
                st.markdown("*None identified.*")

    # Sample feedback
    st.markdown('<div class="section-header"><span>💬</span> Sample Feedback</div>', unsafe_allow_html=True)
    for i, r in teacher_df.head(8).iterrows():
        lbl = r.get("sentiment_label", "Neutral")
        cls = lbl.lower() if isinstance(lbl, str) else "neutral"
        badge_cls = f"badge-{cls}"
        st.markdown(
            f"""<div class="feedback-card {cls}">
              <div class="feedback-text">{r['feedback_text']}</div>
              <div class="feedback-meta">
                <span class="badge {badge_cls}">{lbl}</span>
                {'&nbsp;·&nbsp;' + str(r.get('date', ''))[:10] if 'date' in r and pd.notna(r.get('date')) else ''}
                {'&nbsp;·&nbsp;Rating: ' + str(r.get('rating', '')) if 'rating' in r and pd.notna(r.get('rating')) else ''}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Download
    from utils.report_generator import generate_pdf_report, generate_csv_export
    st.markdown('<div class="section-header"><span>⬇️</span> Export</div>', unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        summary_text = sum_res["summary"] if run_summarizer else "Summarization disabled."
        pdf_score_data = {
            "overall_score": overall_score / 10.0,
            "grade": grade,
            "descriptor": descriptor,
            "aspect_scores": aspect_scores,
            "sentiment_breakdown": {k: round(v / len(teacher_df) * 100, 2) for k, v in counts.items()},
        }
        pdf_bytes = generate_pdf_report(
            teacher_name, pdf_score_data, summary_text, comments[:5]
        )
        if pdf_bytes:
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{teacher_name.replace(' ', '_')}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"pdf_dl_{teacher_name}"
            )
    with dl2:
        csv_bytes = generate_csv_export(teacher_df)
        st.download_button(
            "📊 Download CSV Data",
            data=csv_bytes,
            file_name=f"{teacher_name.replace(' ', '_')}_feedback.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"csv_dl_{teacher_name}"
        )


def render_overview_tab(df_filtered: pd.DataFrame, nlp: dict, run_summarizer: bool):
    from utils.visualizer import (
        sentiment_donut, sentiment_stacked_bar, aspect_frequency_bar,
        teacher_ranking_bar, feedback_volume_bar, aspect_radar
    )
    from utils.aggregator import aggregate_from_scores, cohort_summary

    agg_df = aggregate_from_scores(df_filtered)

    if agg_df.empty:
        st.warning("No data available for aggregation.")
        return

    stats = cohort_summary(agg_df)

    # KPIs
    total_reviews = len(df_filtered)
    counts = df_filtered["sentiment_label"].value_counts().to_dict()
    pos_pct = counts.get("Positive", 0) / total_reviews * 100 if total_reviews > 0 else 0.0
    neu_pct = counts.get("Neutral",  0) / total_reviews * 100 if total_reviews > 0 else 0.0
    neg_pct = counts.get("Negative", 0) / total_reviews * 100 if total_reviews > 0 else 0.0

    overall_score = stats.get("mean_overall", 0.0) * 10
    from nlp.scoring import GRADE_THRESHOLDS
    grade, descriptor = "F", "Poor"
    for threshold, g, d in GRADE_THRESHOLDS:
        if overall_score >= threshold:
            grade, descriptor = g, d
            break

    render_kpi_row(total_reviews, pos_pct, neg_pct, neu_pct, overall_score, grade)

    st.markdown("---")

    # Row 1 (Pie & Volume)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(
            sentiment_donut(counts, "Overall Sentiment Distribution"),
            use_container_width=True,
        )
    with r1c2:
        st.plotly_chart(
            feedback_volume_bar(df_filtered, "Feedback Volume by Teacher"),
            use_container_width=True,
        )

    st.markdown("---")

    # Row 2 (Bar & Radar)
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        visual_ranked = []
        for _, row in agg_df.iterrows():
            visual_ranked.append({
                "teacher": row["teacher_name"],
                "score": row["overall_score"] * 10,
                "grade": row["grade"],
            })
        st.plotly_chart(
            teacher_ranking_bar(visual_ranked),
            use_container_width=True,
        )
    with r2c2:
        cohort_aspects = {
            "Communication": agg_df["communication_score"].mean() * 10 if "communication_score" in agg_df.columns else 50.0,
            "Subject Knowledge": agg_df["subject_knowledge_score"].mean() * 10 if "subject_knowledge_score" in agg_df.columns else 50.0,
            "Engagement": agg_df["engagement_score"].mean() * 10 if "engagement_score" in agg_df.columns else 50.0,
            "Responsiveness": agg_df["responsiveness_score"].mean() * 10 if "responsiveness_score" in agg_df.columns else 50.0,
            "Assignment Quality": agg_df["assignment_quality_score"].mean() * 10 if "assignment_quality_score" in agg_df.columns else 50.0,
        }
        st.plotly_chart(
            aspect_radar(cohort_aspects, "Cohort Average"),
            use_container_width=True,
        )

    st.markdown("---")

    # Table
    st.markdown('<div class="section-header"><span>🏆</span> Teacher Performance Rankings</div>', unsafe_allow_html=True)
    display_cols = [
        "rank", "teacher_name", "subject", "total_reviews",
        "overall_score", "grade", "confidence_band", "strengths", "improvements"
    ]
    display_cols = [c for c in display_cols if c in agg_df.columns]

    table_df = agg_df[display_cols].copy()
    table_df.columns = [c.replace("_", " ").title() for c in table_df.columns]

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
    )

    # Download reports
    from utils.report_generator import generate_csv_export, generate_summary_csv_report
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download CSV Rankings Report",
            data=generate_csv_export(agg_df),
            file_name="teacher_performance_rankings.csv",
            mime="text/csv",
            use_container_width=True,
            key="rankings_csv_download"
        )
    with col_dl2:
        if st.session_state.get("detailed_csv") is None:
            if st.button("📊 Generate Detailed Report (with AI Summaries)", use_container_width=True, key="generate_detailed_csv_btn"):
                with st.spinner("Generating AI summaries for all teachers... This may take a moment."):
                    detailed_csv = generate_summary_csv_report(
                        df_filtered,
                        nlp,
                        run_summarizer=run_summarizer,
                        cache_dict=st.session_state.get("teacher_summaries")
                    )
                    st.session_state["detailed_csv"] = detailed_csv
                st.rerun()
        else:
            st.download_button(
                "⬇️ Download Detailed Report (with AI Summaries)",
                data=st.session_state["detailed_csv"],
                file_name="teacher_performance_detailed_report.csv",
                mime="text/csv",
                use_container_width=True,
                key="detailed_csv_download"
            )
            if st.button("🔄 Clear & Regenerate Report", use_container_width=True, key="regenerate_detailed_csv_btn"):
                st.session_state["detailed_csv"] = None
                st.rerun()


# ---------------------------------------------------------------------------
# Word cloud (optional)
# ---------------------------------------------------------------------------
def render_wordcloud(texts: List[str]):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import io

        combined = " ".join(texts)
        wc = WordCloud(
            width=900,
            height=400,
            background_color="#1a1d2e",
            colormap="cool",
            max_words=120,
            prefer_horizontal=0.85,
        ).generate(combined)

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#1a1d2e")
        ax.set_facecolor("#1a1d2e")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except ImportError:
        st.info("Install `wordcloud` + `matplotlib` to enable word cloud visualisation.")
    except Exception as exc:
        logger.warning(f"Word cloud error: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    init_session_state()


    # Load NLP modules
    nlp = load_nlp_modules()
    if nlp is None:
        st.error("❌ Failed to load NLP modules. Please check your installation.")
        st.code("pip install -r requirements.txt", language="bash")
        return

    # Sidebar (reads dark_mode BEFORE CSS injection)
    source, uploaded_file, manual_text, manual_teacher, run_summarizer, run_aspects, use_fallback, max_entries, analyse_btn, dark_mode = render_sidebar()

    # Inject chosen theme CSS
    _inject_theme(dark_mode)

    # Hero image (base64-encoded for Streamlit compatibility)
    import base64 as _b64
    _hero_path = Path(__file__).parent / "data" / "analytics_hero.png"
    _hero_b64 = ""
    if _hero_path.exists():
        with open(_hero_path, "rb") as _f:
            _hero_b64 = _b64.b64encode(_f.read()).decode()
    _hero_tag = (
        f"<img src='data:image/png;base64,{_hero_b64}' "
        f"class='dashboard-hero-img' alt='AI Analytics illustration'>"
        if _hero_b64 else ""
    )

    # Dashboard header
    st.markdown(
        f"""
        <div class="dashboard-header">
          <div class="dashboard-header-text">
            <h1>🎓 Teacher Feedback Analytics Dashboard</h1>
            <p>AI-powered sentiment analysis, aspect detection &amp; performance scoring for educator feedback</p>
          </div>
          {_hero_tag}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Apply Fast Rule-Based Mode settings
    import os
    if use_fallback:
        os.environ["FORCE_RULE_BASED"] = "1"
    else:
        os.environ["FORCE_RULE_BASED"] = "0"


    # Clear NLP loader caches whenever the setting is changed in the session
    if "prev_fallback" not in st.session_state or st.session_state["prev_fallback"] != use_fallback:
        st.session_state["prev_fallback"] = use_fallback
        from nlp.sentiment import _load_pipeline
        from nlp.aspects import _load_encoder
        from nlp.summarizer import _load_summarizer
        _load_pipeline.cache_clear()
        _load_encoder.cache_clear()
        _load_summarizer.cache_clear()

    # Load data
    df: Optional[pd.DataFrame] = None

    trigger_from_card = st.session_state.get("trigger_analysis", False)
    if trigger_from_card:
        st.session_state["trigger_analysis"] = False

    if analyse_btn or trigger_from_card or st.session_state.get("analysis_done"):
        from utils.data_loader import (
            load_csv, load_excel, generate_sample_dataset,
            validate_dataset,
        )

        if analyse_btn or trigger_from_card:
            st.session_state["analysis_done"] = False
            st.session_state["df"] = None
            st.session_state["scored_df"] = None
            st.session_state["detailed_csv"] = None

        df = st.session_state.get("df")

        if df is None:
            if source == "Use Sample Dataset" or trigger_from_card:
                with st.spinner("Loading sample dataset..."):
                    df = generate_sample_dataset(n_rows=max_entries)
                st.success(f"✅ Loaded sample dataset with **{len(df)}** feedback entries.")

            elif source == "Upload CSV / Excel" and uploaded_file is not None:
                with st.spinner("Parsing uploaded file..."):
                    if uploaded_file.name.endswith(".csv"):
                        df = load_csv(uploaded_file)
                    else:
                        df = load_excel(uploaded_file)
                df = df.head(max_entries)
                st.success(f"✅ Loaded **{len(df)}** feedback entries from `{uploaded_file.name}`.")

            elif source == "Enter Feedback Manually" and manual_text:
                lines = [l.strip() for l in manual_text.strip().splitlines() if l.strip()]
                if not lines:
                    st.warning("Please enter at least one feedback entry.")
                    return
                df = pd.DataFrame({
                    "feedback_text": lines,
                    "teacher_name": [manual_teacher or "Teacher A"] * len(lines),
                })
                df = df.head(max_entries)
                st.success(f"✅ Loaded **{len(df)}** manually entered feedback entries.")

            else:
                st.info("👈 Select a data source in the sidebar and click **Run Analysis**.")
                return

            # Validate
            from utils.data_loader import validate_dataset
            info = validate_dataset(df)
            for warning in info.get("warnings", []):
                st.warning(f"⚠️ {warning}")
            if not info["is_valid"]:
                st.error("❌ Dataset validation failed. Please check your data.")
                return

            st.session_state["df"] = df
            
            # ── Run pipeline and save to scored_df ──
            with st.spinner("Analyzing feedback (running sentiment & aspect models)..."):
                from utils.aggregator import run_pipeline_on_df
                scored_df = run_pipeline_on_df(df)
                st.session_state["scored_df"] = scored_df
            
            st.session_state["analysis_done"] = True
        else:
            scored_df = st.session_state.get("scored_df")

        if scored_df is None or scored_df.empty:
            st.warning("No analyzed data found.")
            return

        # ── Sidebar Filters (Rendered dynamically when scored_df is present) ──
        unique_teachers = sorted(scored_df["teacher_name"].dropna().unique().tolist()) if "teacher_name" in scored_df.columns else []
        unique_subjects = sorted(scored_df["subject"].dropna().unique().tolist()) if "subject" in scored_df.columns else []
        unique_semesters = sorted(scored_df["semester"].dropna().unique().tolist()) if "semester" in scored_df.columns else []

        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🔍 Filter Dashboard")
            selected_teachers = st.multiselect(
                "Teachers",
                options=unique_teachers,
                default=unique_teachers,
                help="Select teachers to include in the analytics",
                on_change=clear_detailed_csv
            )
            selected_subjects = []
            if unique_subjects:
                selected_subjects = st.multiselect(
                    "Subjects",
                    options=unique_subjects,
                    default=unique_subjects,
                    help="Select subjects to include",
                    on_change=clear_detailed_csv
                )
            selected_semesters = []
            if unique_semesters:
                selected_semesters = st.multiselect(
                    "Semesters",
                    options=unique_semesters,
                    default=unique_semesters,
                    help="Select semesters to include",
                    on_change=clear_detailed_csv
                )

        # Apply filters to scored_df to produce df_filtered
        df_filtered = scored_df.copy()
        if "teacher_name" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["teacher_name"].isin(selected_teachers)]

        if "subject" in df_filtered.columns and unique_subjects:
            df_filtered = df_filtered[df_filtered["subject"].isin(selected_subjects)]

        if "semester" in df_filtered.columns and unique_semesters:
            df_filtered = df_filtered[df_filtered["semester"].isin(selected_semesters)]

        if df_filtered.empty:
            st.warning("⚠️ No feedback matches the selected filters. Adjust your filters in the sidebar.")
            return

        # ── Tabs ──
        has_teachers = "teacher_name" in df_filtered.columns and df_filtered["teacher_name"].nunique() > 1
        tab_labels = ["📊 Overview", "👩‍🏫 By Teacher", "☁️ Word Cloud", "📋 Raw Data"]
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            render_overview_tab(df_filtered, nlp, run_summarizer)

        with tabs[1]:
            st.markdown('<div class="section-header"><span>👩‍🏫</span> Individual Teacher Analysis</div>', unsafe_allow_html=True)
            if has_teachers:
                teachers = sorted(df_filtered["teacher_name"].unique().tolist())
                selected = st.selectbox("Select a teacher:", teachers, key="teacher_select")
                if selected:
                    render_teacher_detail(selected, nlp, df_filtered[df_filtered["teacher_name"] == selected], run_summarizer)
            else:
                teacher_name = df_filtered.get("teacher_name", pd.Series(["Teacher A"])).iloc[0] if "teacher_name" in df_filtered.columns else "Teacher"
                render_teacher_detail(
                    teacher_name,
                    nlp, df_filtered, run_summarizer
                )

        with tabs[2]:
            st.markdown('<div class="section-header"><span>☁️</span> Feedback Word Cloud</div>', unsafe_allow_html=True)
            texts = df_filtered["feedback_text"].tolist()
            if st.button("🔄 Generate Word Cloud"):
                render_wordcloud(texts)

        with tabs[3]:
            st.markdown('<div class="section-header"><span>📋</span> Raw Dataset</div>', unsafe_allow_html=True)
            st.dataframe(df_filtered, use_container_width=True, height=500)
            from utils.report_generator import generate_csv_export
            st.download_button(
                "⬇️ Download Full Dataset (CSV)",
                data=generate_csv_export(df_filtered),
                file_name="teacher_feedback_full.csv",
                mime="text/csv",
            )

    else:
        # Landing state
        st.markdown(
            """
            <div style='text-align:center; padding: 2rem 1rem 1rem 1rem;'>
              <div style='font-size:4rem; margin-bottom:0.5rem;'>🚀</div>
              <h2 style='color:#a78bfa; font-size:1.6rem;'>Ready to Analyse</h2>
              <p style='color:#64748b; max-width:550px; margin:0 auto 1.5rem auto; line-height:1.7;'>
                Choose a data source in the sidebar, configure your options,
                and click <strong style='color:#6c63ff;'>Run Analysis</strong>. Or, tap any of the cards below to test immediately with the sample dataset.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Feature cards
        features = [
            ("😊", "Sentiment Analysis", "Classify feedback as Positive, Negative, or Neutral using RoBERTa"),
            ("🎯", "Aspect Detection", "Identify key pedagogical dimensions using sentence-transformers"),
            ("🏆", "Performance Scoring", "Compute weighted 0–100 scores with letter grades"),
            ("✍️", "AI Summarization", "Generate concise feedback summaries using BART"),
            ("📈", "Trend Analysis", "Visualize score changes over time with interactive charts"),
            ("📄", "PDF Reports", "Export professional performance reports for each teacher"),
        ]
        cols = st.columns(3)
        for i, (icon, title, desc) in enumerate(features):
            with cols[i % 3]:
                st.markdown(
                    f"""<div class="kpi-card" style='text-align:left; margin-bottom:0.5rem; height: 165px;'>
                      <div style='font-size:1.8rem; margin-bottom:0.3rem;'>{icon}</div>
                      <div style='font-weight:700; color:#e2e8f0; margin-bottom:0.2rem;'>{title}</div>
                      <div style='font-size:0.8rem; color:#64748b; line-height:1.4;'>{desc}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button(f"⚡ Test {title}", key=f"feat_{i}", use_container_width=True):
                    st.session_state["dataset_source"] = "sample"
                    st.session_state["trigger_analysis"] = True
                    st.rerun()


if __name__ == "__main__":
    main()
