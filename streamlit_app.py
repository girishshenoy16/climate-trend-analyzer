"""
Climate Trend Analyzer - Streamlit Interactive Dashboard.

Secondary deployment target for Streamlit Cloud.
Provides interactive analytics, Plotly charts, and scenario simulation.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import DOCS_DATA_DIR, PROCESSED_DIR
from src.constants import (
    COL_DATE,
    COL_HUMIDITY,
    COL_MONTH,
    COL_PRECIPITATION,
    COL_SOLAR_RADIATION,
    COL_TEMP_MEAN,
    COL_WIND_SPEED,
    RISK_COLORS,
)

st.set_page_config(
    page_title="Climate Trend Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Consistent Color Palette ────────────────────────────────────────────────

COLORS = {
    "temperature": "#E67E22",
    "temperature_fill": "rgba(230, 126, 34, 0.15)",
    "rainfall": "#3498DB",
    "rainfall_fill": "rgba(52, 152, 219, 0.15)",
    "forecast": "#F1C40F",
    "forecast_fill": "rgba(241, 196, 15, 0.15)",
    "anomaly": "#E74C3C",
    "normal": "rgba(52, 152, 219, 0.3)",
    "risk_low": "#2ECC71",
    "risk_moderate": "#F1C40F",
    "risk_high": "#E67E22",
    "risk_very_high": "#E74C3C",
    "bg_card": "rgba(26, 41, 64, 0.8)",
    "text": "#FFFFFF",
    "text_secondary": "#A0B4C8",
    "accent": "#3498DB",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "danger": "#E74C3C",
}

# ─── Plotly Layout (Executive Style) ─────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,27,45,0)",
    plot_bgcolor="rgba(15,27,45,0)",
    font=dict(color="#A0B4C8", family="Inter, sans-serif", size=12),
    margin=dict(l=50, r=30, t=60, b=50),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.1)",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
    hoverlabel=dict(
        bgcolor="rgba(26,41,64,0.95)",
        font_size=12,
        font_family="Inter, sans-serif",
    ),
)

# ─── Custom CSS (Executive Dashboard) ────────────────────────────────────────

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 960px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Typography */
    h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.5px;
        text-align: center;
    }

    h2 {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #A0B4C8 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* KPI Cards - Uniform Styling */
    .stMetric {
        background: linear-gradient(135deg, rgba(26,41,64,0.7), rgba(15,27,45,0.7));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        transition: all 0.2s ease;
    }

    .stMetric:hover {
        border-color: rgba(52,152,219,0.3);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    .stMetric label {
        font-size: 0.68rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #A0B4C8 !important;
        font-weight: 500;
    }

    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700;
        color: #FFFFFF !important;
        line-height: 1.3;
        text-align: center;
    }

    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        font-weight: 500;
    }

    /* Section Divider */
    .section-divider {
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1.8rem auto;
        max-width: 960px;
    }

    /* Insight Cards */
    .insight-card {
        background: linear-gradient(135deg, rgba(26,41,64,0.6), rgba(15,27,45,0.6));
        border-left: 3px solid #3498DB;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        font-size: 0.9rem;
        line-height: 1.6;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        min-height: 48px;
    }

    /* Recommendation Cards */
    .rec-card {
        background: linear-gradient(135deg, rgba(26,41,64,0.6), rgba(15,27,45,0.6));
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.85rem;
        font-size: 0.9rem;
        min-height: 48px;
    }

    /* Priority Badges */
    .priority-badge {
        font-size: 0.6rem;
        font-weight: 600;
        padding: 4px 11px;
        border-radius: 12px;
        white-space: nowrap;
        flex-shrink: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        line-height: 1.4;
        min-width: 68px;
        text-align: center;
        display: inline-block;
    }

    .priority-high {
        background: rgba(231,76,60,0.15);
        color: #E74C3C;
        border: 1px solid rgba(231,76,60,0.3);
    }

    .priority-medium {
        background: rgba(230,126,34,0.15);
        color: #E67E22;
        border: 1px solid rgba(230,126,34,0.3);
    }

    .priority-low {
        background: rgba(46,204,113,0.15);
        color: #2ECC71;
        border: 1px solid rgba(46,204,113,0.3);
    }

    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: #8FA3B8;
        font-size: 0.7rem;
        margin-top: 2.75rem;
        padding-top: 1.4rem;
        padding-bottom: 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        line-height: 1.7;
        max-width: 960px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,27,45,1), rgba(10,18,30,1));
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.2rem !important;
        margin-bottom: 0.25rem !important;
    }

    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.35rem;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        padding: 0.55rem 0.8rem;
        border-radius: 8px;
        transition: background 0.2s ease;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255,255,255,0.05);
    }

    [data-testid="stSidebar"] .stRadio > div > div[data-checked="true"] > label {
        background: rgba(52,152,219,0.15);
        border-left: 3px solid #3498DB;
        font-weight: 500;
        box-shadow: inset 0 0 12px rgba(52,152,219,0.08);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.6rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #3498DB;
    }

    /* DataFrame Styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Chart Container */
    .stPlotlyChart {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Page Subtitle */
    .page-subtitle {
        color: #A0B4C8;
        font-size: 0.9rem;
        font-weight: 400;
        margin-top: -0.5rem;
        margin-bottom: 1.25rem;
        line-height: 1.5;
        text-align: center;
    }

    /* Date Range Chip */
    .date-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, rgba(26,41,64,0.7), rgba(15,27,45,0.7));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.45rem 0.9rem;
        font-size: 0.78rem;
        color: #A0B4C8;
        margin-bottom: 0.25rem;
    }

    .date-chip strong {
        color: #FFFFFF;
        font-weight: 600;
    }

    /* Observation Card */
    .observation-card {
        background: linear-gradient(135deg, rgba(26,41,64,0.6), rgba(15,27,45,0.6));
        border-left: 3px solid #3498DB;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1.25rem;
        font-size: 0.88rem;
        line-height: 1.65;
        color: #A0B4C8;
    }

    .observation-card strong {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_processed_data():
    path = PROCESSED_DIR / "climate_daily_processed.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=[COL_DATE])
    return df


def load_json_feed(filename):
    path = DOCS_DATA_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🌍 Climate Trend Analyzer")
    st.markdown("**v1.0.0** — Executive Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Executive Overview", "Temperature Analysis", "Precipitation Analysis",
         "Humidity & Solar", "Forecast & Projections", "Anomaly Detection",
         "Monthly Distribution", "Regional Map"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #6B8299; line-height: 1.6;">
        <strong>Data Source</strong><br>
        NASA POWER & Open-Meteo APIs<br><br>
        <strong>Location</strong><br>
        New Delhi, India<br>
        28.61°N, 77.21°E
    </div>
    """, unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────

df = load_processed_data()
exec_summary = load_json_feed("executive_summary.json")
forecast_data = load_json_feed("forecast.json")
anomaly_data = load_json_feed("anomalies.json")
regional_data = load_json_feed("regional_map.json")

if df is None:
    st.error("Processed data not found. Run `python main.py` first.")
    st.stop()

# ─── Helper: Last Updated ─────────────────────────────────────────────────────


# ─── Executive Overview ───────────────────────────────────────────────────────

if page == "Executive Overview":

    st.title("Executive Overview")
    st.markdown(
        '<p class="page-subtitle">Comprehensive Climate Risk Assessment and Key Performance Indicators (2015–2024)</p>',
        unsafe_allow_html=True,
    )

    kpis = exec_summary.get("kpis", {}) if exec_summary else {}

    risk_cat = kpis.get("risk_category", "Unknown")
    risk_score = kpis.get("risk_score", 0)
    period = f"{kpis.get('analysis_start_year', 'N/A')}–{kpis.get('analysis_end_year', 'N/A')}"
    station = kpis.get("station_name", "the selected region")
    p_val = kpis.get("historical_trend_p_value", 1.0)
    trend_val = kpis.get("warming_rate_per_decade", 0)
    if p_val < 0.05:
        trend_desc = f"a statistically significant trend of {trend_val:+.2f} °C/decade"
    else:
        trend_desc = (
            f"a slight long-term temperature trend of {trend_val:+.2f} °C/decade; "
            "however, the observed trend is <strong>not statistically significant</strong> "
            f"based on the available data (p = {p_val:.2f})"
        )
    overview = (
        f"The analysis of {station} over {period} identified {trend_desc}, "
        f"with an average temperature of {kpis.get('avg_temperature', 0):.1f} °C. "
        f"Climate risk is assessed as <strong>{risk_cat}</strong> "
        f"(score: {risk_score:.2f}), driven by temperature trends and "
        f"{kpis.get('anomaly_days', 0)} detected anomaly days "
        f"({kpis.get('anomaly_percentage', 0):.1f}% of the period). "
        f"The 3-year forecast projects a trend of "
        f"{kpis.get('forecast_trend_per_decade', 0):.2f} °C/decade, "
        f"consistent with the historical pattern."
    )
    st.markdown(f'<div style="max-width: 960px; line-height: 1.7;">{overview}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # KPI Cards — 5 columns with uniform spacing
    trend_delta = "Not Statistically Significant" if kpis.get("historical_trend_p_value", 1.0) >= 0.05 else ("Increasing" if kpis.get("warming_rate_per_decade", 0) > 0 else "Decreasing")
    trend_delta_color = "off" if kpis.get("historical_trend_p_value", 1.0) >= 0.05 else "inverse"
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1:
        st.metric("Avg Temperature", f"{kpis.get('avg_temperature', 0):.1f} °C", delta="Increasing", delta_color="inverse")
    with c2:
        st.metric("Temperature Trend", f"{kpis.get('warming_rate_per_decade', 0):+.2f} °C/dec", delta=trend_delta, delta_color=trend_delta_color)
    with c3:
        st.metric("Avg Precipitation", f"{kpis.get('avg_precipitation', 0):.1f} mm/day", delta="Variable", delta_color="off")
    with c4:
        st.metric("Anomaly Days", f"{kpis.get('anomaly_days', 0)}", delta=f"{kpis.get('anomaly_percentage', 0):.1f}% of obs", delta_color="inverse")
    with c5:
        st.metric("Forecast Trend", f"{kpis.get('forecast_trend_per_decade', 0):+.2f} °C/dec", delta="Stable Projection", delta_color="inverse")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Climate Risk Card — Enlarged score, centered badge+score
    risk_color = RISK_COLORS.get(risk_cat, "#FFFFFF")
    risk_interp = {"Low": "Minimal Climate Risk", "Moderate": "Moderate Climate Risk", "High": "Elevated Climate Risk", "Very High": "Severe Climate Risk"}.get(risk_cat, "Climate Risk")
    risk_tooltip = "Composite indicator derived from normalized climate metrics: temperature trend, rainfall deviation, anomaly frequency, and forecast consistency. This is an analytical index, not a direct scientific measurement."
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(26,41,64,0.9), rgba(15,27,45,0.9));
                border-left: 5px solid {risk_color}; border-radius: 12px; padding: 1.5rem 1.75rem; margin: 1.25rem 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2); text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">⚠️</span>
            <h3 style="color: {risk_color}; margin: 0; font-size: 1.25rem; font-weight: 600;">Climate Risk Assessment</h3>
            <span title="{risk_tooltip}" style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: rgba(255,255,255,0.15); color: #A0B4C8; font-size: 0.75rem; cursor: help; font-style: italic;">ℹ️</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: center; gap: 3rem; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; justify-content: center;">
                <span class="risk-badge" style="background: {risk_color}22; color: {risk_color}; border: 1px solid {risk_color}44; padding: 8px 18px;">{risk_cat}</span>
            </div>
            <div style="display: flex; flex-direction: column; align-items: center;">
                <span style="font-size: 2.4rem; font-weight: 700; color: white; line-height: 1;">{risk_score:.2f}</span>
                <span style="color: #A0B4C8; font-size: 0.8rem; margin-top: 0.4rem;">{risk_interp}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Insights with icons — uniform height, centered text
    if exec_summary.get("insights"):
        st.markdown("## 💡 Key Insights")
        icons = ["🌡️", "🌧️", "⚠️", "📈"]
        for i, insight in enumerate(exec_summary["insights"]):
            icon = icons[i % len(icons)]
            st.markdown(f"""
            <div class="insight-card">
                <span style="font-size: 1.1rem; margin-right: 0.6rem; flex-shrink: 0;">{icon}</span>
                <span style="color: #A0B4C8;">{insight}</span>
            </div>
            """, unsafe_allow_html=True)

    # Strategic Recommendations — grouped by priority tier
    if exec_summary.get("recommendations"):
        st.markdown("## 🎯 Strategic Recommendations")
        recs = exec_summary["recommendations"]
        # Tier assignments based on recommendation content
        immediate = [recs[i] for i in [0]] if len(recs) > 0 else []
        medium = [recs[i] for i in [1, 3, 4] if i < len(recs)]
        long_term = [recs[i] for i in [2, 5, 6] if i < len(recs)]

        def _render_rec_section(title, items, badge_label, badge_cls):
            if not items:
                return
            st.markdown(f"**{title}**")
            for rec in items:
                st.markdown(f"""
                <div class="rec-card">
                    <span class="priority-badge {badge_cls}">{badge_label}</span>
                    <span style="color: #A0B4C8;">{rec}</span>
                </div>
                """, unsafe_allow_html=True)

        _render_rec_section("Immediate Actions", immediate, "High", "priority-high")
        _render_rec_section("Medium-Term Actions", medium, "Medium", "priority-medium")
        _render_rec_section("Long-Term Actions", long_term, "Low", "priority-low")

# ─── Temperature Analysis ─────────────────────────────────────────────────────

elif page == "Temperature Analysis":

    st.title("Temperature Trend Analysis")
    st.markdown(
        '<p class="page-subtitle">Historical Temperature Patterns and Long-Term Trend Analysis (2015–2024)</p>',
        unsafe_allow_html=True,
    )

    date_range = st.slider(
        "Select Date Range",
        min_value=df[COL_DATE].min().to_pydatetime(),
        max_value=df[COL_DATE].max().to_pydatetime(),
        value=(df[COL_DATE].min().to_pydatetime(), df[COL_DATE].max().to_pydatetime()),
    )

    mask = (df[COL_DATE] >= pd.Timestamp(date_range[0])) & (df[COL_DATE] <= pd.Timestamp(date_range[1]))
    filtered = df[mask]

    st.markdown(
        f'<div class="date-chip">📅 <strong>{pd.Timestamp(date_range[0]).strftime("%d %b %Y")}</strong> — <strong>{pd.Timestamp(date_range[1]).strftime("%d %b %Y")}</strong></div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered[COL_DATE], y=filtered[COL_TEMP_MEAN],
        mode="lines", name="Daily Temperature",
        line=dict(color=COLORS["temperature"], width=1),
        opacity=0.4,
    ))

    ma30 = filtered[COL_TEMP_MEAN].rolling(30, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=filtered[COL_DATE], y=ma30,
        mode="lines", name="30-Day Moving Avg",
        line=dict(color="#E74C3C", width=2.5),
    ))

    fig.update_layout(
        title="Historical Temperature Trend",
        xaxis_title="Date", yaxis_title="Temperature (°C)",
        height=490, **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(title_font_size=13, tickfont_size=11)
    fig.update_yaxes(title_font_size=13, tickfont_size=11)
    st.plotly_chart(fig, use_container_width=True)

    mean_t = filtered[COL_TEMP_MEAN].mean()
    max_t = filtered[COL_TEMP_MEAN].max()
    min_t = filtered[COL_TEMP_MEAN].min()
    std_t = filtered[COL_TEMP_MEAN].std()
    temp_range = max_t - min_t

    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        st.metric("🌡️ Mean Temperature", f"{mean_t:.2f} °C")
    with col2:
        st.metric("🔺 Maximum Temperature", f"{max_t:.2f} °C")
    with col3:
        st.metric("🔻 Minimum Temperature", f"{min_t:.2f} °C")
    with col4:
        st.metric("📊 Standard Deviation", f"{std_t:.2f} °C")

    st.markdown(f"""
    <div class="observation-card">
        <strong>Key Observation:</strong> The station records a seasonal temperature range of
        <strong>{temp_range:.1f} °C</strong> (from {min_t:.1f} °C to {max_t:.1f} °C), with a mean of
        <strong>{mean_t:.1f} °C</strong> and standard deviation of <strong>{std_t:.1f} °C</strong>,
        indicating a {('wide' if std_t > 5 else 'moderate' if std_t > 3 else 'narrow')} seasonal
        variation consistent with the regional climate pattern.
    </div>
    """, unsafe_allow_html=True)

# ─── Precipitation Analysis ───────────────────────────────────────────────────

elif page == "Precipitation Analysis":

    st.title("Precipitation Trend Analysis")
    st.markdown(
        '<p class="page-subtitle">Historical Rainfall Distribution and Precipitation Characteristics (2015–2024)</p>',
        unsafe_allow_html=True,
    )

    if COL_MONTH in df.columns:
        monthly_precip = df.groupby(COL_MONTH)[COL_PRECIPITATION].sum()
        month_labels = [["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"][i-1] for i in monthly_precip.index]

        fig = px.bar(
            x=month_labels,
            y=monthly_precip.values,
            labels={"x": "Month", "y": "Total Precipitation (mm)"},
            title="Monthly Precipitation Totals",
            color=monthly_precip.values,
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=490, **PLOTLY_LAYOUT)
        fig.update_xaxes(title_font_size=13, tickfont_size=11)
        fig.update_yaxes(title_font_size=13, tickfont_size=11)
        fig.update_layout(coloraxis_colorbar=dict(title="Monthly Rainfall (mm)"))

        top3 = monthly_precip.nlargest(3)
        for idx, val in zip(top3.index, top3.values):
            fig.add_annotation(
                x=month_labels[idx - 1], y=val,
                text=f"{val:.0f}", showarrow=False,
                yshift=12, font=dict(size=11, color="#FFFFFF", family="Inter, sans-serif"),
            )

        st.plotly_chart(fig, use_container_width=True)

    avg_p = df[COL_PRECIPITATION].mean()
    max_p = df[COL_PRECIPITATION].max()
    wet_days = int((df[COL_PRECIPITATION] > 1).sum())
    peak_month = month_labels[monthly_precip.idxmax() - 1]
    peak_monthly = monthly_precip.max()

    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.metric("🌧️ Avg Daily Precipitation", f"{avg_p:.2f} mm")
    with col2:
        st.metric("🌦️ Max Daily Precipitation", f"{max_p:.2f} mm")
    with col3:
        st.metric("📅 Wet Days (>1 mm)", f"{wet_days:,}")

    st.markdown(f"""
    <div class="observation-card">
        <strong>Key Observation:</strong> Precipitation is heavily concentrated in the monsoon months
        (Jun–Sep), which account for the majority of annual rainfall. The wettest month is
        <strong>{peak_month}</strong> with <strong>{peak_monthly:.0f} mm</strong> total, while the
        long-term daily average is <strong>{avg_p:.2f} mm</strong>, indicating a distinct wet–dry
        seasonal cycle characteristic of the Delhi climate.
    </div>
    """, unsafe_allow_html=True)

# ─── Humidity & Solar Radiation ────────────────────────────────────────────────

elif page == "Humidity & Solar":

    st.title("Humidity & Solar Radiation Analysis")
    st.markdown(
        '<p class="page-subtitle">Long-Term Atmospheric Moisture and Solar Energy Pattern Analysis (2015–2024)</p>',
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["💧 Humidity", "☀️ Solar Radiation"])

    with tab1:
        st.markdown("### Relative Humidity Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[COL_DATE], y=df[COL_HUMIDITY],
            mode="lines", name="Daily Humidity",
            line=dict(color="#1ABC9C", width=1), opacity=0.4,
        ))
        ma30 = df[COL_HUMIDITY].rolling(30, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df[COL_DATE], y=ma30,
            mode="lines", name="30-Day Moving Avg",
            line=dict(color="#E74C3C", width=2.5),
        ))
        fig.update_layout(
            title="Relative Humidity (2015–2024)",
            xaxis_title="Date", yaxis_title="Humidity (%)",
            height=490, **PLOTLY_LAYOUT,
        )
        fig.update_xaxes(title_font_size=13, tickfont_size=11)
        fig.update_yaxes(title_font_size=13, tickfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

        mean_h = df[COL_HUMIDITY].mean()
        max_h = df[COL_HUMIDITY].max()
        min_h = df[COL_HUMIDITY].min()
        std_h = df[COL_HUMIDITY].std()

        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            st.metric("💧 Mean Humidity", f"{mean_h:.1f}%")
        with c2:
            st.metric("📈 Maximum Humidity", f"{max_h:.1f}%")
        with c3:
            st.metric("📉 Minimum Humidity", f"{min_h:.1f}%")

        st.markdown(f"""
        <div class="observation-card">
            <strong>Key Observation:</strong> Relative humidity ranges from <strong>{min_h:.1f}%</strong>
            to <strong>{max_h:.1f}%</strong> with a mean of <strong>{mean_h:.1f}%</strong> and standard
            deviation of <strong>{std_h:.1f}%</strong>, reflecting
            {('significant' if std_h > 15 else 'moderate' if std_h > 10 else 'mild')} moisture
            variability across the {('monsoon-influenced' if mean_h > 55 else 'semi-arid')} climate regime.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Solar Radiation Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[COL_DATE], y=df[COL_SOLAR_RADIATION],
            mode="lines", name="Daily Solar Radiation",
            line=dict(color="#F39C12", width=1), opacity=0.4,
        ))
        ma30 = df[COL_SOLAR_RADIATION].rolling(30, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df[COL_DATE], y=ma30,
            mode="lines", name="30-Day Moving Avg",
            line=dict(color="#E74C3C", width=2.5),
        ))
        fig.update_layout(
            title="Solar Radiation (2015–2024)",
            xaxis_title="Date", yaxis_title="Solar Radiation (MJ/m²/day)",
            height=490, **PLOTLY_LAYOUT,
        )
        fig.update_xaxes(title_font_size=13, tickfont_size=11)
        fig.update_yaxes(title_font_size=13, tickfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

        mean_s = df[COL_SOLAR_RADIATION].mean()
        max_s = df[COL_SOLAR_RADIATION].max()
        min_s = df[COL_SOLAR_RADIATION].min()

        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            st.metric("☀️ Mean Radiation", f"{mean_s:.2f} MJ/m²/day")
        with c2:
            st.metric("🌞 Maximum Radiation", f"{max_s:.2f} MJ/m²/day")
        with c3:
            st.metric("🌅 Minimum Radiation", f"{min_s:.2f} MJ/m²/day")

        st.markdown(f"""
        <div class="observation-card">
            <strong>Key Observation:</strong> Solar radiation varies between <strong>{min_s:.2f}</strong>
            and <strong>{max_s:.2f} MJ/m²/day</strong> with a mean of <strong>{mean_s:.2f} MJ/m²/day</strong>,
            indicating {('high' if max_s > 22 else 'moderate' if max_s > 18 else 'low')} peak
            intensity and {('wide' if (max_s - min_s) > 10 else 'moderate' if (max_s - min_s) > 6 else 'narrow')}
            seasonal variability driven by cloud cover and day-length cycles.
        </div>
        """, unsafe_allow_html=True)

# ─── Forecast & Projections ───────────────────────────────────────────────────

elif page == "Forecast & Projections":

    st.title("3-Year Temperature Forecast")
    st.markdown(
        '<p class="page-subtitle">Holt-Winters Exponential Smoothing Model Projection and Reliability Assessment (2025–2027)</p>',
        unsafe_allow_html=True,
    )

    if forecast_data:
        meta = forecast_data.get("metadata", {})
        forecast_entries = forecast_data.get("data", [])
        reliability = meta.get("reliability_label", "N/A")
        trend_dec = meta.get("trend_per_decade", 0)
        r2 = meta.get("trend_r_squared", 0)
        horizon = len(forecast_entries)
        fvals = [d["forecast"] for d in forecast_entries]
        forecast_mean = sum(fvals) / len(fvals) if fvals else 0

        r2_label = ("Strong fit" if r2 > 0.8 else "Moderate fit" if r2 > 0.5 else "Weak fit")

        st.markdown(
            f'<div style="max-width: 960px; line-height: 1.7; color: #A0B4C8; margin-bottom: 0.5rem;">'
            f'A <strong style="color:#FFFFFF;">{horizon}-day forecast</strong> using the Holt-Winters model '
            f'projects a trend of <strong style="color:#FFFFFF;">{trend_dec:.2f} °C/decade</strong> '
            f'with <strong style="color:#FFFFFF;">{reliability}</strong> reliability '
            f'(R² = <strong style="color:#FFFFFF;">{r2:.4f}</strong>, {r2_label}). '
            f'The forecast mean temperature is <strong style="color:#FFFFFF;">{forecast_mean:.1f} °C</strong>.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.metric("📅 Forecast Horizon", f"{horizon} days")
        with c2:
            st.metric("🌡️ Expected Mean", f"{forecast_mean:.1f} °C")
        with c3:
            st.metric("📈 Trend", f"{trend_dec:.2f} °C/decade")
        with c4:
            st.metric("🎯 Confidence", reliability)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df[COL_DATE], y=df[COL_TEMP_MEAN],
            mode="lines", name="Historical",
            line=dict(color=COLORS["temperature"], width=1.5),
            opacity=0.6,
        ))

        fdates = [str(d["date"]) for d in forecast_entries]
        fupper = [d["forecast_upper"] for d in forecast_entries]
        flower = [d["forecast_lower"] for d in forecast_entries]

        fig.add_trace(go.Scatter(
            x=fdates + fdates[::-1],
            y=fupper + flower[::-1],
            fill="toself", fillcolor="rgba(241,196,15,0.2)",
            line=dict(width=0), name="95% Confidence Interval",
        ))

        fig.add_trace(go.Scatter(
            x=fdates, y=fvals,
            mode="lines", name="Forecast",
            line=dict(color=COLORS["forecast"], width=2.5, dash="dash"),
        ))

        forecast_start_date = forecast_entries[0]["date"]
        fig.add_shape(
            type="line", x0=forecast_start_date, x1=forecast_start_date,
            y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color="rgba(241,196,15,0.5)", width=1),
        )
        fig.add_annotation(
            x=forecast_start_date, y=1, yref="paper",
            text="Forecast Start", showarrow=False,
            font=dict(color=COLORS["forecast"], size=11),
            xanchor="left", yanchor="bottom",
        )

        fig.update_layout(
            title="Holt-Winters Temperature Forecast (3-Year Projection)",
            xaxis_title="Date", yaxis_title="Temperature (°C)",
            height=490, **PLOTLY_LAYOUT,
        )
        fig.update_xaxes(title_font_size=13, tickfont_size=11)
        fig.update_yaxes(title_font_size=13, tickfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

        if meta:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("## 📊 Forecast Model Metadata")

            c1, c2, c3, c4 = st.columns(4, gap="small")
            with c1:
                st.metric("🛡️ Reliability", reliability)
            with c2:
                st.metric("📊 Reliability Score", f"{meta.get('reliability_score', 0):.3f}")
            with c3:
                st.metric("📈 Trend", f"{trend_dec:.3f} °C/dec")
            with c4:
                st.metric("📉 R²", f"{r2:.4f}", help=r2_label)

            reasons = meta.get("classification_reasons", [])
            if reasons:
                st.warning(f"**Classification:** {meta.get('forecast_class', 'N/A').replace('_', ' ').title()}")
                for r in reasons:
                    st.caption(r)

            action = meta.get("recommended_action", "")
            if action:
                st.info(f"**Recommendation:** {action}")

            st.markdown(f"""
            <div class="observation-card">
                <strong>Key Forecast Insights:</strong>
                <ul style="margin: 0.5rem 0 0 1.2rem; padding: 0; line-height: 1.8; color: #A0B4C8;">
                    <li>The model projects a <strong>{trend_dec:.2f} °C/decade</strong> warming trend with <strong>{r2_label}</strong> explanatory power (R² = {r2:.4f}).</li>
                    <li>Forecast reliability is rated <strong>{reliability}</strong> based on walk-forward validation across 3 test folds.</li>
                    <li>The 95% confidence interval widens over the projection horizon, reflecting increasing uncertainty at longer ranges.</li>
                    <li>The {horizon}-day forecast spans from <strong>{min(fvals):.1f} °C</strong> to <strong>{max(fvals):.1f} °C</strong>, within the historical variability range.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Forecast data not available. Run pipeline first.")

# ─── Anomaly Detection ────────────────────────────────────────────────────────

elif page == "Anomaly Detection":

    st.title("Climate Anomaly Detection")
    st.markdown(
        '<p class="page-subtitle">Statistical Temperature Deviation Analysis and Extreme Event Identification (2015–2024)</p>',
        unsafe_allow_html=True,
    )

    if anomaly_data:
        anomaly_dates = [pd.Timestamp(d["date"]) for d in anomaly_data]
        anomaly_temps = [d.get("temperature") for d in anomaly_data]
        anomaly_pct = len(anomaly_data) / len(df) * 100
        normal_count = len(df) - len(anomaly_data)
        anomaly_max = max(anomaly_temps) if anomaly_temps else 0
        anomaly_min = min(anomaly_temps) if anomaly_temps else 0
        largest_anomaly = max(abs(anomaly_max - df[COL_TEMP_MEAN].mean()),
                              abs(anomaly_min - df[COL_TEMP_MEAN].mean()))

        st.markdown(
            f'<div style="max-width: 960px; line-height: 1.7; color: #A0B4C8; margin-bottom: 0.5rem;">'
            f'A total of <strong style="color:#FFFFFF;">{len(anomaly_data):,} anomaly days</strong> '
            f'(<strong style="color:#FFFFFF;">{anomaly_pct:.1f}%</strong> of all observations) were detected '
            f'across the analysis period, indicating {("frequent" if anomaly_pct > 5 else "occasional" if anomaly_pct > 2 else "rare")} '
            f'extreme temperature deviations from the historical norm.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df[COL_DATE], y=df[COL_TEMP_MEAN],
            mode="lines", name="Normal Temperature",
            line=dict(color=COLORS["normal"], width=1),
            opacity=0.4,
        ))

        fig.add_trace(go.Scatter(
            x=anomaly_dates, y=anomaly_temps,
            mode="markers", name="Detected Climate Anomalies",
            marker=dict(color=COLORS["anomaly"], size=5, opacity=0.85,
                        line=dict(width=1, color="white")),
        ))

        fig.update_layout(
            title="Anomaly Detection Results",
            xaxis_title="Date", yaxis_title="Temperature (°C)",
            height=490, **PLOTLY_LAYOUT,
        )
        fig.update_xaxes(title_font_size=13, tickfont_size=11)
        fig.update_yaxes(title_font_size=13, tickfont_size=11)
        fig.update_layout(
            modebar=dict(bgcolor="rgba(0,0,0,0)", orientation="v"),
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.metric("⚠️ Anomaly Days", f"{len(anomaly_data):,}")
        with c2:
            st.metric("📊 Anomaly Rate", f"{anomaly_pct:.1f}%")
        with c3:
            st.metric("✅ Normal Observations", f"{normal_count:,}")
        with c4:
            st.metric("📏 Largest Deviation", f"{largest_anomaly:.2f} °C")

        st.markdown(f"""
        <div class="observation-card">
            <strong>Key Findings:</strong>
            <ul style="margin: 0.5rem 0 0 1.2rem; padding: 0; line-height: 1.8; color: #A0B4C8;">
                <li><strong>{len(anomaly_data):,} days</strong> exceeded the statistical threshold, representing <strong>{anomaly_pct:.1f}%</strong> of the record.</li>
                <li>The largest single deviation reached <strong>{largest_anomaly:.2f} °C</strong> above or below the long-term mean.</li>
                <li>With <strong>{normal_count:,}</strong> normal observations, the climate remains predominantly stable with localized extreme events.</li>
                <li>The anomaly pattern suggests {("increasing frequency of extreme events consistent with warming trends" if anomaly_pct > 4 else "intermittent extreme events within a stable climate regime")}.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No anomalies detected.")

# ─── Monthly Distribution ─────────────────────────────────────────────────────

elif page == "Monthly Distribution":

    st.title("Monthly Climate Distribution")
    st.markdown(
        '<p class="page-subtitle">Seasonal Temperature Pattern and Monthly Variability Analysis (2015–2024)</p>',
        unsafe_allow_html=True,
    )

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if COL_MONTH in df.columns:
        monthly_temp = df.groupby(COL_MONTH)[COL_TEMP_MEAN].agg(["mean", "min", "max", "std"])

        warmest_idx = monthly_temp["mean"].idxmax()
        coldest_idx = monthly_temp["mean"].idxmin()
        warmest_month = month_names[warmest_idx - 1]
        coldest_month = month_names[coldest_idx - 1]
        warmest_temp = monthly_temp["mean"].max()
        coldest_temp = monthly_temp["mean"].min()
        annual_mean = monthly_temp["mean"].mean()
        max_std = monthly_temp["std"].max()
        max_std_month = month_names[monthly_temp["std"].idxmax() - 1]
        seasonal_range = warmest_temp - coldest_temp

        st.markdown(
            f'<div style="max-width: 960px; line-height: 1.7; color: #A0B4C8; margin-bottom: 0.5rem;">'
            f'The warmest month is <strong style="color:#FFFFFF;">{warmest_month}</strong> '
            f'(<strong style="color:#FFFFFF;">{warmest_temp:.1f} °C</strong>), while the coldest is '
            f'<strong style="color:#FFFFFF;">{coldest_month}</strong> '
            f'(<strong style="color:#FFFFFF;">{coldest_temp:.1f} °C</strong>), yielding a seasonal range of '
            f'<strong style="color:#FFFFFF;">{seasonal_range:.1f} °C</strong>. '
            f'Annual mean temperature is <strong style="color:#FFFFFF;">{annual_mean:.1f} °C</strong> '
            f'with highest monthly variability in <strong style="color:#FFFFFF;">{max_std_month}</strong>.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.metric("🔥 Warmest Month", f"{warmest_month} ({warmest_temp:.1f} °C)")
        with c2:
            st.metric("❄️ Coldest Month", f"{coldest_month} ({coldest_temp:.1f} °C)")
        with c3:
            st.metric("📊 Annual Mean", f"{annual_mean:.1f} °C")
        with c4:
            st.metric("📈 Highest Std Dev", f"{max_std:.2f} °C ({max_std_month})")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[month_names[i-1] for i in monthly_temp.index],
            y=monthly_temp["mean"],
            name="Mean",
            marker_color="#E67E22",
            error_y=dict(type="data", array=monthly_temp["std"], visible=True),
        ))
        fig.update_layout(
            title="Average Temperature by Month (with Std Dev)",
            xaxis_title="Month", yaxis_title="Temperature (°C)",
            height=490, **PLOTLY_LAYOUT,
        )
        fig.update_xaxes(title_font_size=13, tickfont_size=11)
        fig.update_yaxes(title_font_size=13, tickfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 📋 Monthly Statistics")
        stats_df = pd.DataFrame({
            "Month": month_names,
            "Mean (°C)": monthly_temp["mean"].values,
            "Min (°C)": monthly_temp["min"].values,
            "Max (°C)": monthly_temp["max"].values,
            "Std Dev": monthly_temp["std"].values,
        })

        def highlight_extremes(s):
            is_max = s == s.max()
            is_min = s == s.min()
            return ["background-color: rgba(231,76,60,0.15); color: #E74C3C" if v else
                    "background-color: rgba(52,152,219,0.15); color: #3498DB" if w else ""
                    for v, w in zip(is_max, is_min)]

        styled = stats_df.style.format({
            "Mean (°C)": "{:.2f}", "Min (°C)": "{:.2f}",
            "Max (°C)": "{:.2f}", "Std Dev": "{:.2f}"
        }).apply(highlight_extremes, subset=["Mean (°C)"])

        st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

        st.markdown(f"""
        <div class="observation-card">
            <strong>Key Seasonal Insights:</strong>
            <ul style="margin: 0.5rem 0 0 1.2rem; padding: 0; line-height: 1.8; color: #A0B4C8;">
                <li><strong>{warmest_month}</strong> is the warmest month at <strong>{warmest_temp:.1f} °C</strong>, while <strong>{coldest_month}</strong> is the coldest at <strong>{coldest_temp:.1f} °C</strong>.</li>
                <li>The seasonal temperature range of <strong>{seasonal_range:.1f} °C</strong> indicates a {('wide' if seasonal_range > 15 else 'moderate' if seasonal_range > 8 else 'narrow')} annual cycle.</li>
                <li>Monthly variability peaks in <strong>{max_std_month}</strong> (σ = {max_std:.2f} °C), suggesting {('unstable transitional weather' if max_std > 5 else 'relatively stable conditions')} during that period.</li>
                <li>The annual mean of <strong>{annual_mean:.1f} °C</strong> is consistent with a {('tropical' if annual_mean > 25 else 'subtropical' if annual_mean > 20 else 'temperate')} climate classification.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ─── Regional Map ─────────────────────────────────────────────────────────────

elif page == "Regional Map":

    st.title("Regional Climate Station Map")
    st.markdown("*Interactive Geographic View of Climate Monitoring Location (2015–2024)*", unsafe_allow_html=False)

    if regional_data and regional_data.get("stations"):
        station = regional_data["stations"][0]

        # Map with increased visual impact
        st.map(pd.DataFrame({"lat": [station["lat"]], "lon": [station["lon"]]}), zoom=5)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # KPI Cards with icons and formatted values
        avg_temp = station.get('avg_temp', 0)
        total_precip = station.get('total_precip', 0)
        anomaly_days = station.get('anomaly_days', 0)
        risk_category = station.get('risk_category', 'N/A')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌡 Avg Temperature", f"{avg_temp:.2f} °C")
        with col2:
            st.metric("🌧 Total Precipitation", f"{total_precip:,.1f} mm")
        with col3:
            st.metric("⚠ Anomaly Days", f"{anomaly_days:,}")
        with col4:
            st.metric("🛡 Risk Level", risk_category)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Context Panel
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(26,41,64,0.6), rgba(15,27,45,0.6));
                    border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
                    padding: 1.25rem 1.5rem; margin: 0.5rem 0;">
            <h3 style="color: #A0B4C8; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">
                Station Information
            </h3>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1.5rem;">
                <div>
                    <div style="color: #6B8299; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;">Data Source</div>
                    <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 500;">NASA POWER & Open-Meteo</div>
                </div>
                <div>
                    <div style="color: #6B8299; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;">Location</div>
                    <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 500;">{station.get('name', 'New Delhi, India')}</div>
                </div>
                <div>
                    <div style="color: #6B8299; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;">Coordinates</div>
                    <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 500;">{station.get('lat', 28.61):.2f}°N, {station.get('lon', 77.21):.2f}°E</div>
                </div>
                <div>
                    <div style="color: #6B8299; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;">Analysis Period</div>
                    <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 500;">2015–2024</div>
                </div>
                <div>
                    <div style="color: #6B8299; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;">Risk Classification</div>
                    <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 500;">{risk_category}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Regional data not available.")

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer-note">
    <strong>Data Sources:</strong> NASA POWER API | Open-Meteo Climate API<br>
    Climate Trend Analyzer — Automated Climate Analysis & Forecasting System<br>
    Last Updated: 25 July 2026, 20:46 UTC
</div>
""", unsafe_allow_html=True)
