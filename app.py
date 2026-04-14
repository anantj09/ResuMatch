"""
app.py
------
Resume & Job Description Matcher — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys, os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.extractor import extract_text
from src.scorer import score as _score
from src.similarity import get_model

@st.cache_resource(show_spinner="Loading SBERT model...")
def load_model():
    """Load once, reuse forever across all reruns."""
    return get_model()

@st.cache_data(show_spinner=False)
def run_analysis(resume_text: str, jd_text: str) -> dict:
    import time
    model = load_model()
    print(f"[1] model ready")
    t0 = time.time()
    result = _score(resume_text, jd_text, model)
    print(f"[2] score() done in {time.time()-t0:.2f}s")
    return result

# Page config
st.set_page_config(
    page_title="ResuMatch · NLP Resume Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "history" not in st.session_state:
    st.session_state["history"] = []

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #FFFFFF;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-size: 1.05rem;
    color: #6b7280;
    margin-bottom: 2rem;
}

.score-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 20px 60px rgba(26,26,46,0.3);
}

.score-number {
    font-family: 'DM Serif Display', serif;
    font-size: 5rem;
    font-weight: 400;
    line-height: 1;
    color: white;
}

.score-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #a0aec0;
    margin-top: 0.5rem;
}

.verdict-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin-top: 1rem;
    text-transform: uppercase;
}

.verdict-strong { background: #d1fae5; color: #065f46; }
.verdict-good   { background: #dbeafe; color: #1e40af; }
.verdict-fair   { background: #fef3c7; color: #92400e; }
.verdict-weak   { background: #fee2e2; color: #991b1b; }

.kw-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
}

.kw-matched { background: #d1fae5; color: #065f46; }
.kw-missing { background: #fee2e2; color: #991b1b; }
.kw-extra   { background: #e0e7ff; color: #3730a3; }

.section-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}

.stTextArea textarea {
    border-radius: 12px !important;
    border: 1.5px solid #e5e7eb !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}

.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed #d1d5db;
    border-radius: 14px;
    padding: 1rem;
    transition: border-color 0.2s;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.7rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}

.divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}

.tip-box {
    background: #f0f9ff;
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #374151;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# Helpers

def verdict(score: float) -> tuple[str, str]:
    if score >= 75:
        return "Strong Match", "verdict-strong"
    elif score >= 55:
        return "Good Match", "verdict-good"
    elif score >= 35:
        return "Fair Match", "verdict-fair"
    else:
        return "Weak Match", "verdict-weak"


def gauge_chart(value: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 36, "family": "DM Serif Display"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
            "bar": {"color": "#6366f1", "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#fee2e2"},
                {"range": [35, 55], "color": "#fef3c7"},
                {"range": [55, 75], "color": "#dbeafe"},
                {"range": [75, 100],"color": "#d1fae5"},
            ],
            "threshold": {
                "line": {"color": "#374151", "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans"},
    )
    return fig


def section_bar_chart(section_scores: dict) -> go.Figure:
    labels, values, colors = [], [], []
    color_map = {
        "skills":     "#6366f1",
        "experience": "#8b5cf6",
        "education":  "#a78bfa",
    }
    for k, v in section_scores.items():
        if v is not None:
            labels.append(k.capitalize())
            values.append(round(v * 100, 1))
            colors.append(color_map.get(k, "#6366f1"))

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont={"size": 13, "family": "DM Sans"},
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 110], showgrid=True, gridcolor="#f3f4f6", ticksuffix="%"),
        xaxis=dict(showgrid=False),
        font={"family": "DM Sans"},
        showlegend=False,
    )
    return fig


def keyword_donut(gap: dict) -> go.Figure:
    matched = len(gap["matched"])
    missing = len(gap["missing"])
    fig = go.Figure(go.Pie(
        labels=["Matched", "Missing"],
        values=[matched, missing],
        hole=0.6,
        marker_colors=["#6366f1", "#fca5a5"],
        textinfo="label+percent",
        textfont={"size": 12, "family": "DM Sans"},
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[{
            "text": f"{matched}/{matched+missing}",
            "x": 0.5, "y": 0.5,
            "font_size": 18,
            "font_family": "DM Serif Display",
            "showarrow": False,
        }],
    )
    return fig


def chips_html(keywords: list[str], cls: str) -> str:
    if not keywords:
        return "<span style='color:#9ca3af; font-size:0.85rem;'>None found</span>"
    return " ".join(
        f'<span class="kw-chip {cls}">{kw}</span>' for kw in keywords
    )

def score_color(s):
    if s >= 75: return "#059669"
    if s >= 55: return "#2563eb"
    if s >= 35: return "#d97706"
    return "#dc2626"

def history_bar_chart(history):
    names  = [h["label"] for h in history]
    scores = [h["overall_score"] for h in history]
    colors = [score_color(s) for s in scores]
    fig = go.Figure(go.Bar(
        x=names, y=scores, marker_color=colors, marker_line_width=0,
        text=[f"{s}%" for s in scores], textposition="outside",
        textfont={"size": 13, "family": "DM Sans"},
    ))
    fig.update_layout(
        height=280, margin=dict(t=20,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,110], showgrid=True, gridcolor="#f3f4f6", ticksuffix="%"),
        xaxis=dict(showgrid=False), font={"family":"DM Sans"}, showlegend=False,
    )
    return fig

# Header
st.markdown("""
<div style="padding: 2rem 0 1rem;">
    <div class="hero-title">📄 ResuMatch</div>
    <div class="hero-sub">
        Semantic resume & job description matcher
    </div>
</div>
""", unsafe_allow_html=True)

# Load sample data shortcut (must run BEFORE widgets are instantiated)
with st.expander("🧪 Try with sample data"):
    if st.button("Load sample resume & JD"):
        with open("data/samples/sample_resume.txt") as f:
            st.session_state["_resume_val"] = f.read()
        with open("data/samples/sample_jd.txt") as f:
            st.session_state["_jd_val"] = f.read()
        st.rerun()

# Input Section
col_name, col_clr = st.columns([2, 1])
with col_name:
    resume_label = st.text_input("📌 Label this resume (for history)", value="Resume 1", key="resume_label")
with col_clr:
    st.markdown("<div style='margin-top:1.8rem'>", unsafe_allow_html=True)
    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
col_r, col_j = st.columns(2, gap="large")

with col_r:
    st.markdown("#### 🧑‍💼 Resume")
    resume_file = st.file_uploader(
        "Upload resume", type=["pdf", "docx", "txt"],
        key="resume_upload", label_visibility="collapsed"
    )
    resume_text_input = st.text_area(
        "Or paste resume text",
        height=220,
        placeholder="Paste your resume here...",
        value=st.session_state.get("_resume_val", ""),
        key="resume_text",
    )

with col_j:
    st.markdown("#### 💼 Job Description")
    jd_file = st.file_uploader(
        "Upload JD", type=["pdf", "docx", "txt"],
        key="jd_upload", label_visibility="collapsed"
    )
    jd_text_input = st.text_area(
        "Or paste job description text",
        height=220,
        placeholder="Paste the job description here...",
        value=st.session_state.get("_jd_val", ""),
        key="jd_text",
    )

st.markdown("<br>", unsafe_allow_html=True)
col_btn, _ = st.columns([1, 3])
with col_btn:
    analyze_btn = st.button("⚡ Analyze Match", use_container_width=True)

# Analysis
if analyze_btn:
    # Resolve text sources
    resume_text = ""
    jd_text = ""

    if resume_file:
        resume_text = extract_text(resume_file)
    elif resume_text_input.strip():
        resume_text = resume_text_input.strip()

    if jd_file:
        jd_text = extract_text(jd_file)
    elif jd_text_input.strip():
        jd_text = jd_text_input.strip()

    if not resume_text or not jd_text:
        st.warning("⚠️ Please provide both a resume and a job description.")
        st.stop()

    with st.spinner("🔍 Running semantic analysis…"):
        result = run_analysis(resume_text, jd_text)

    # Duplicate label check
    existing_labels = [h["label"] for h in st.session_state["history"]]
    current_label = resume_label.strip() or f"Resume {len(existing_labels)+1}"
    if current_label in existing_labels:
        st.warning(f"⚠️ **\"{current_label}\"** already exists in history. Rename it above before analyzing.")
        st.stop()

    # Save to history
    st.session_state["history"].append({
        **result,
        "label": current_label,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("## 📊 Match Results")

    # Row 1: Score + Gauge + Section bars
    c1, c2, c3 = st.columns([1.2, 1.5, 1.5], gap="large")

    overall = result["overall_score"]
    v_label, v_cls = verdict(overall)

    with c1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-number">{overall}</div>
            <div class="score-label">Overall Match Score</div>
            <div><span class="verdict-badge {v_cls}">{v_label}</span></div>
            <div style="margin-top:1.2rem; font-size:0.82rem; color:#cbd5e1;">
                Semantic · Keyword · Section weighted
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("**Semantic Similarity Gauge**")
        st.plotly_chart(
            gauge_chart(round(result["semantic_score"] * 100, 1)),
            use_container_width=True, config={"displayModeBar": False}
        )

    with c3:
        sec = result["section_scores"]
        valid = {k: v for k, v in sec.items() if v is not None}
        if valid:
            st.markdown("**Section-wise Scores**")
            st.plotly_chart(
                section_bar_chart(valid),
                use_container_width=True, config={"displayModeBar": False}
            )
        else:
            st.info("Section headers not detected — using full-text mode.")

    # Row 2: Keyword gap
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("## 🔑 Keyword Gap Analysis")

    gap = result["keyword_gap"]
    kc1, kc2 = st.columns([1, 2], gap="large")

    with kc1:
        st.markdown("**Keyword Coverage**")
        st.plotly_chart(
            keyword_donut(gap),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown(f"""
        <div class="tip-box">
            <b>Coverage rate:</b> {round(gap['match_rate'] * 100, 1)}% of JD keywords found in resume
        </div>
        """, unsafe_allow_html=True)

    with kc2:
        st.markdown("**✅ Matched Keywords**")
        st.markdown(chips_html(gap["matched"], "kw-matched"), unsafe_allow_html=True)

        st.markdown("<br>**❌ Missing from Resume** *(add these to improve score)*", unsafe_allow_html=True)
        st.markdown(chips_html(gap["missing"], "kw-missing"), unsafe_allow_html=True)

        st.markdown("<br>**➕ Extra Skills in Resume** *(not in JD but noteworthy)*", unsafe_allow_html=True)
        st.markdown(chips_html(gap["extra"][:10], "kw-extra"), unsafe_allow_html=True)

    # Row 3: Section text preview
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("## 🔎 Section Breakdown")

    for section in ("skills", "experience", "education"):
        res_sec = result["resume_sections"].get(section, "").strip()
        jd_sec  = result["jd_sections"].get(section, "").strip()
        sec_score = result["section_scores"].get(section)

        label = f"{'🛠' if section=='skills' else '💼' if section=='experience' else '🎓'} **{section.capitalize()}**"
        score_badge = f" — `{round(sec_score*100,1)}% match`" if sec_score is not None else " — *not detected*"

        with st.expander(f"{label}{score_badge}"):
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("**Resume**")
                st.text(res_sec[:800] + ("…" if len(res_sec) > 800 else "") if res_sec else "Not found")
            with s2:
                st.markdown("**Job Description**")
                st.text(jd_sec[:800] + ("…" if len(jd_sec) > 800 else "") if jd_sec else "Not found")

    # Footer
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#9ca3af; font-size:0.82rem; padding-bottom:2rem;">
        ResuMatch | Make your Resume aligned for the job you want
    </div>
    """, unsafe_allow_html=True)

# History panel
history = st.session_state["history"]
if history:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("## 📋 Resume Comparison History")

    st.plotly_chart(history_bar_chart(history), use_container_width=True, config={"displayModeBar": False})

    rows = []
    for h in history:
        v_lbl, _ = verdict(h["overall_score"])
        rows.append({
            "Label":     h["label"],
            "Time":      h["timestamp"],
            "Score":     f"{h['overall_score']}%",
            "Verdict":   v_lbl,
            "Semantic":  f"{round(h['semantic_score']*100,1)}%",
            "Keywords":  f"{round(h['keyword_gap']['match_rate']*100,1)}%",
            "Skills":    f"{round(h['section_scores'].get('skills',0)*100,1)}%" if h['section_scores'].get('skills') else "—",
            "Exp.":      f"{round(h['section_scores'].get('experience',0)*100,1)}%" if h['section_scores'].get('experience') else "—",
            "Edu.":      f"{round(h['section_scores'].get('education',0)*100,1)}%" if h['section_scores'].get('education') else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    csv = pd.DataFrame(rows).to_csv(index=False)
    st.download_button("⬇ Export history as CSV", csv, "resumatch_history.csv", "text/csv")