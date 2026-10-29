"""
ChargeWise Dashboard | TEB3123 Machine Learning | Group 10
Run:  streamlit run app.py
Keep every file from the dashboard folder (including .streamlit/config.toml) next to this script.
"""
import json
from datetime import date, time, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="ChargeWise", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- design tokens
INK, MUTED, LINE = "#1D1D1F", "#6E6E73", "#E6E8F0"
ACCENT, ACCENT_SOFT, GREY = "#4F63E8", "#AEB9F5", "#C9CDD8"
# Apple font stack: real SF Pro on Mac / iPhone, Inter (closest free match) elsewhere
FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", Arial, sans-serif'

CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
.stApp, .stApp p, .stApp div, .stApp label, .stApp button, .stApp input, .stApp li, .stApp td, .stApp th {{
  font-family: {FONT} !important; -webkit-font-smoothing: antialiased; }}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: "Material Symbols Rounded" !important; }}
.stApp {{ background: #FBFBFD; color: {INK}; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stDecoration"] {{ display: none; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ max-width: 1180px; padding-top: 1.2rem; padding-bottom: 2rem; }}

/* ---------- top navigation ---------- */
.brand {{ font-size: 22px; font-weight: 700; letter-spacing: -0.03em; color: {INK}; padding-top: 6px; }}
.brand span {{ color: {ACCENT}; }}
.st-key-nav [role="radiogroup"] {{ gap: 2px; justify-content: center; flex-wrap: nowrap; }}
.st-key-nav [data-testid="stRadioOption"], .st-key-nav label[data-baseweb="radio"] {{ padding: 7px 13px; border-radius: 999px; margin: 0; white-space: nowrap;
  cursor: pointer; transition: background .2s; }}
.st-key-nav [data-testid="stRadioOption"]:hover {{ background: #EEF0F6; }}
.st-key-nav [data-testid="stRadioOption"] > div > div:first-child, .st-key-nav label[data-baseweb="radio"] > div:first-child {{ display: none; }}
.st-key-nav [data-testid="stRadioOption"] p {{ font-size: 13px; font-weight: 500; color: {INK}; letter-spacing: .03em; text-transform: uppercase; }}
.st-key-nav [data-testid="stRadioOption"][data-selected="true"], .st-key-nav label[data-baseweb="radio"]:has(input:checked) {{ background: {INK}; }}
.st-key-nav [data-testid="stRadioOption"][data-selected="true"] p, .st-key-nav label:has(input:checked) p {{ color: #fff; }}
.pill-dark {{ display: inline-block; float: right; background: {INK}; color: #fff; border-radius: 999px; padding: 8px 18px;
  font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; margin-top: 4px; }}

/* ---------- hero ---------- */
.st-key-hero {{ background: radial-gradient(circle at 85% 20%, #FFFFFF 0%, rgba(255,255,255,0) 40%),
  linear-gradient(135deg, #F5F6FF 0%, #E4E8FF 55%, #CDD5FF 100%); border-radius: 32px; padding: 52px 48px 44px; margin-top: 10px; }}
.kicker-pill {{ display: inline-block; background: rgba(255,255,255,.75); border: 1px solid #DDE1F7; border-radius: 999px;
  padding: 6px 14px; font-size: 12px; font-weight: 600; color: {ACCENT}; letter-spacing: .04em; text-transform: uppercase; }}
.hero-title {{ font-size: 58px; line-height: 1.04; font-weight: 700; letter-spacing: -0.035em; color: {INK}; margin: 18px 0 16px; }}
.hero-sub {{ font-size: 17px; line-height: 1.55; color: {MUTED}; max-width: 470px; margin-bottom: 18px; }}
.glass {{ background: rgba(255,255,255,.62); backdrop-filter: blur(14px); border: 1px solid rgba(255,255,255,.9);
  border-radius: 22px; padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 10px 30px rgba(79,99,232,.10); }}
.glass .g-label {{ font-size: 13px; font-weight: 600; color: {INK}; }}
.glass .g-value {{ font-size: 40px; font-weight: 700; letter-spacing: -0.03em; color: {INK}; line-height: 1.1; }}
.glass .g-note {{ font-size: 12px; color: {MUTED}; }}

/* ---------- stats strip ---------- */
.stats {{ display: grid; grid-template-columns: repeat(var(--cols, 5), 1fr); gap: 8px; padding: 34px 6px 10px; text-align: center; }}
.stats .v {{ font-size: 34px; font-weight: 700; letter-spacing: -0.03em; color: {INK}; }}
.stats .l {{ font-size: 13px; color: {MUTED}; margin-top: 2px; }}

/* ---------- section headings ---------- */
.sec {{ text-align: center; margin: 64px auto 26px; max-width: 720px; }}
.sec .k {{ font-size: 30px; font-weight: 600; letter-spacing: -0.025em; color: {INK}; }}
.sec .t {{ font-size: 40px; font-weight: 700; letter-spacing: -0.035em; color: {INK}; line-height: 1.1; }}
.sec .s {{ font-size: 16px; color: {MUTED}; margin-top: 14px; line-height: 1.6; }}

/* ---------- cards ---------- */
[class*="st-key-card"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 26px; padding: 18px 20px;
  box-shadow: 0 6px 24px rgba(17,24,39,.05); }}
.step {{ padding: 14px 6px 22px; }}
.step .n {{ font-size: 30px; font-weight: 700; letter-spacing: -0.03em; color: {INK}; }}
.step .h {{ font-size: 16px; font-weight: 600; color: {INK}; margin: 2px 0 6px; }}
.step .d {{ font-size: 14px; color: {MUTED}; line-height: 1.55; }}
.right {{ text-align: right; }}
.charger {{ background: linear-gradient(180deg, #F5F6FF, #DCE2FF); border-radius: 160px; height: 100%; min-height: 470px;
  display: flex; align-items: center; justify-content: center; }}

.plan {{ border-radius: 26px; padding: 26px 24px; border: 1px solid {LINE}; background: linear-gradient(180deg,#FFFFFF,#F1F3FF); height: 100%; }}
.plan.hot {{ background: linear-gradient(180deg,#E9ECFF,#CBD3FF); border: 2px solid {ACCENT}; }}
.plan .name {{ font-size: 18px; font-weight: 600; color: {INK}; text-align: center; }}
.plan .big {{ font-size: 44px; font-weight: 700; letter-spacing: -0.03em; text-align: center; color: {INK}; margin-top: 8px; }}
.plan .big small {{ font-size: 16px; font-weight: 500; color: {MUTED}; }}
.plan .sub {{ text-align: center; font-size: 13px; color: {MUTED}; margin-bottom: 14px; }}
.plan ul {{ list-style: none; padding: 0; margin: 0; }}
.plan li {{ font-size: 14px; color: {INK}; padding: 7px 0; border-top: 1px solid rgba(29,29,31,.07); }}
.plan li b {{ float: right; }}
.badge {{ display: inline-block; background: {ACCENT}; color: #fff; font-size: 11px; font-weight: 600; border-radius: 999px;
  padding: 3px 10px; letter-spacing: .04em; }}

.metric-grid {{ display: grid; grid-template-columns: repeat(var(--cols, 4), 1fr); gap: 14px; margin: 18px 0; }}
.mcard {{ background: #fff; border: 1px solid {LINE}; border-radius: 22px; padding: 18px 20px; }}
.mcard .l {{ font-size: 13px; color: {MUTED}; }}
.mcard .v {{ font-size: 32px; font-weight: 700; letter-spacing: -0.03em; color: {INK}; }}
.mcard .n {{ font-size: 12px; color: {MUTED}; }}
.banner {{ border-radius: 22px; padding: 18px 22px; font-size: 15px; line-height: 1.55; margin: 8px 0 4px; }}
.banner.ok {{ background: #EAF7EF; color: #17663A; }}
.banner.warn {{ background: #FFF6E5; color: #8A5A00; }}
.banner.bad {{ background: #FDECEC; color: #A12A2A; }}
.banner.info {{ background: linear-gradient(135deg,#EEF0FF,#DCE2FF); color: {INK}; }}
.cta {{ border-radius: 32px; padding: 44px 44px; margin-top: 60px; color: {INK};
  background: radial-gradient(circle at 90% 30%, #FFFFFF 0%, rgba(255,255,255,0) 45%), linear-gradient(135deg,#E9ECFF,#C9D1FF); }}
.cta .t {{ font-size: 36px; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1; }}
.cta .s {{ font-size: 16px; color: {MUTED}; margin-top: 10px; max-width: 640px; line-height: 1.6; }}

/* ---------- widgets ---------- */
.stButton button, .stFormSubmitButton button {{ border-radius: 999px !important; background: {INK} !important; color: #fff !important;
  border: none !important; padding: 10px 24px !important; font-weight: 600 !important; letter-spacing: .04em; }}
.stButton button p, .stFormSubmitButton button p {{ color: #fff !important; font-size: 13px; text-transform: uppercase; }}
.stButton button:hover, .stFormSubmitButton button:hover {{ background: #3A3A3D !important; }}
[data-testid="stForm"] {{ border: 1px solid {LINE}; border-radius: 26px; background: #fff; padding: 26px; }}
[data-baseweb="select"] > div, [data-baseweb="input"], .stNumberInput input, .stDateInput input, .stTimeInput input {{ border-radius: 14px !important; }}
[data-testid="stExpander"] details {{ border: none; border-bottom: 1px solid {LINE}; border-radius: 0; background: transparent; }}
[data-testid="stExpander"] summary p {{ font-size: 16px; font-weight: 600; color: {INK}; }}
[data-testid="stDataFrame"] {{ border-radius: 16px; overflow: hidden; }}

/* ---------- footer ---------- */
.footer {{ text-align: center; border-top: 1px solid {LINE}; margin-top: 70px; padding: 34px 0 10px; }}
.footer .b {{ font-size: 22px; font-weight: 700; letter-spacing: -0.03em; }}
.footer .links {{ font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; margin: 10px 0 16px; }}
.footer .links span {{ margin: 0 12px; white-space: nowrap; }}
.footer .small {{ font-size: 12px; color: {MUTED}; }}

@media (max-width: 800px) {{
  .hero-title {{ font-size: 40px; }} .st-key-hero {{ padding: 32px 22px; }}
  .stats, .metric-grid {{ grid-template-columns: repeat(2, 1fr); row-gap: 22px; }} .stats .v {{ font-size: 28px; }}
  .st-key-nav [role="radiogroup"] {{ flex-wrap: wrap; }} .pill-dark {{ display: none; }} .charger {{ min-height: 300px; }}
  .sec .t {{ font-size: 30px; }} .sec .k {{ font-size: 22px; }} .right {{ text-align: left; }}
}}
</style>
"""


# ---------------------------------------------------------------- helpers
def html(s):
    """Render HTML without Markdown turning indented lines into code blocks."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


KEEP = {"kWh", "vs", "÷"}


def tc(text):
    """Capitalise Each Word, but keep units such as kWh untouched."""
    words = []
    for w in str(text).split(" "):
        if w and w[0].islower() and w not in KEEP:
            w = w[0].upper() + w[1:]
        words.append("Vs" if w == "vs" else w)
    return " ".join(words)


FEATURE_LABELS = {
    "kwh_requested": "kWh Requested", "miles_requested": "Distance Requested (km)", "parking_hours": "Parking Hours",
    "request_hour": "Request Hour", "day_of_week": "Day Of Week", "month": "Month", "is_weekend": "Is Weekend",
    "paid_period": "Paid Period", "driver_past_sessions": "Driver Past Sessions",
    "driver_past_avg_kwh": "Driver Past Average kWh", "driver_past_ratio": "Driver Past Delivered ÷ Requested",
    "station": "Station", "vehicle_model": "Vehicle Model",
}


def style(fig, title, height=360, legend=False):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01, y=0.96, font=dict(size=17, color=INK, family=FONT)),
        font=dict(family=FONT, color=MUTED, size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=58, b=8), height=height, showlegend=legend, bargap=0.25,
        hoverlabel=dict(font_family=FONT, bgcolor="#fff", bordercolor=LINE),
        legend=dict(orientation="h", y=-0.2, x=0, title_text=""),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="", tickangle=0, title_font=dict(color=MUTED, size=13))
    fig.update_yaxes(gridcolor="#EEF0F6", zeroline=False, linecolor="rgba(0,0,0,0)", title_font=dict(color=MUTED, size=13))
    return fig


def chart(fig, key):
    with st.container(key=f"card_{key}"):
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def section(kicker, title, sub=""):
    html(f'<div class="sec"><div class="k">{kicker}</div><div class="t">{title}</div>'
         + (f'<div class="s">{sub}</div>' if sub else "") + "</div>")


def stats(items):
    cells = "".join(f'<div><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items)
    html(f'<div class="stats" style="--cols:{len(items)}">{cells}</div>')


def metric_cards(items):
    cells = "".join(f'<div class="mcard"><div class="l">{l}</div><div class="v">{v}</div><div class="n">{n}</div></div>'
                    for l, v, n in items)
    html(f'<div class="metric-grid" style="--cols:{len(items)}">{cells}</div>')


def table(df):
    out = df.copy()
    out.columns = [tc(c).replace("R2", "R²") for c in out.columns]
    for c in out.columns:
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(tc)
    st.dataframe(out, hide_index=True, width="stretch")


CHARGER_SVG = """
<svg viewBox="0 0 170 330" width="170" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="body" x1="0" x2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#E3E7FF"/></linearGradient></defs>
<rect x="34" y="8" width="102" height="270" rx="44" fill="url(#body)" stroke="#C9D1FF" stroke-width="2"/>
<rect x="52" y="40" width="66" height="92" rx="18" fill="#1D1D1F"/>
<text x="85" y="80" text-anchor="middle" font-size="13" fill="#AEB9F5" font-family="-apple-system,Inter,sans-serif">kWh</text>
<text x="85" y="108" text-anchor="middle" font-size="26" font-weight="700" fill="#FFFFFF" font-family="-apple-system,Inter,sans-serif">11.1</text>
<rect x="74" y="152" width="22" height="46" rx="10" fill="#4F63E8"/>
<path d="M85 198 C 85 262, 150 250, 150 322" stroke="#1D1D1F" stroke-width="7" fill="none" stroke-linecap="round"/>
<rect x="44" y="276" width="82" height="16" rx="8" fill="#AEB9F5"/>
</svg>
"""


# ---------------------------------------------------------------- data
@st.cache_resource
def load_model():
    return joblib.load(HERE / "model.joblib")


@st.cache_data
def load_data():
    read = lambda f: pd.read_csv(HERE / f)
    meta = json.loads((HERE / "meta.json").read_text())
    data = {k: read(f"{f}.csv") for k, f in [("results", "results"), ("tuning", "tuning"), ("covid", "covid_results"),
                                              ("cleaning", "cleaning_log"), ("importance", "feature_importance"),
                                              ("preds", "test_predictions"), ("clean", "clean_data")]}
    data["clean"]["request_entry_time"] = pd.to_datetime(data["clean"]["request_entry_time"])
    return meta, data


model = load_model()
meta, D = load_data()
BEST = meta["best_model"]
TARGET = "energy_charged"
PRED = f"pred_{BEST}"
C, R, P = D["clean"], D["results"], D["preds"]
over_request = P["kwh_requested"].sum() / P[TARGET].sum() - 1
# kWh requested = miles x the car's efficiency (how NREL's system calculated it), learned from the data
_ratio = C["kwh_requested"] / C["miles_requested"]
KWH_PER_MILE = _ratio.groupby(C["vehicle_model"]).median().round(3).to_dict()
KWH_PER_MILE_DEFAULT = round(float(_ratio.median()), 3)
KM_PER_MILE = 1.609  # the model was trained in miles; the form uses km and converts behind the scenes
flex_share = ((P["parking_hours"] - P[PRED] / 6.6) >= 1).mean()

# ---------------------------------------------------------------- top bar
html(CSS)
PAGES = ["Home", "Model Results", "Demand Drivers", "Predict", "Smart Charging"]
if "nav" not in st.session_state:
    st.session_state.nav = "Home"


def go_to(page):
    st.session_state.nav = page


nav_l, nav_m, nav_r = st.columns([1, 4.2, 0.9], vertical_alignment="center")
with nav_l:
    html('<div class="brand">Charge<span>Wise</span></div>')
with nav_m:
    page = st.radio("Navigation", PAGES, horizontal=True, key="nav", label_visibility="collapsed")
with nav_r:
    html('<div class="pill-dark">Group 10</div>')


def footer():
    html(f"""
    <div class="footer">
      <div class="b">Charge<span style="color:{ACCENT}">Wise</span></div>
      <div class="links"><span>Home</span><span>Model Results</span><span>Demand Drivers</span><span>Predict</span><span>Smart Charging</span></div>
      <div class="small">Data: NREL Workplace Charging Data, Colorado, USA (2016 To 2021). Insights For UTP Are Illustrative And Need Local Validation.</div>
      <div class="small" style="margin-top:6px">TEB3123 Machine Learning · Group 10 · Universiti Teknologi PETRONAS</div>
    </div>""")


# ================================================================ HOME
if page == "Home":
    with st.container(key="hero"):
        left, right = st.columns([1.25, 1], gap="large", vertical_alignment="center")
        with left:
            html(f"""
            <span class="kicker-pill">TEB3123 Machine Learning · Group 10</span>
            <div class="hero-title">Predicting The<br>Next Charge.</div>
            <div class="hero-sub">ChargeWise predicts how much electricity each EV charging session will really use,
            using only what the driver enters when they plug in. Better forecasts mean smarter, greener campus charging.</div>""")
            st.button("Try A Prediction", on_click=go_to, args=("Predict",))
        with right:
            pre = D["covid"].query("Scenario == 'Pre-COVID only'").iloc[0]
            html(f"""
            <div class="glass"><div class="g-label">Over-Requested Energy</div>
              <div class="g-value">+{over_request:.0%}</div>
              <div class="g-note">Drivers ask for far more energy than they actually take</div></div>
            <div class="glass"><div class="g-label">Model Accuracy (Pre-COVID)</div>
              <div class="g-value">R² {pre['R2']:.2f}</div>
              <div class="g-note">{BEST}, average error {pre['MAE']:.1f} kWh per session</div></div>
            <div class="glass"><div class="g-label">Flexible Sessions</div>
              <div class="g-value">{flex_share:.0%}</div>
              <div class="g-note">Have at least 1 spare hour to shift charging (6.6 kW)</div></div>""")

    stats([(f"{len(C):,}", "Clean Sessions"), (f"{C['driverId'].nunique():,}", "Unique Drivers"),
           (f"{C['station'].nunique()}", "Charging Stations"), (f"{C[TARGET].mean():.1f} kWh", "Average Delivered"),
           (f"{(C[TARGET] / C['kwh_requested']).median():.0%}", "Delivered ÷ Requested")])

    section("About The Data:", "The Story Of Our Dataset.",
            "Real workplace charging records from the National Renewable Energy Laboratory (NREL) in Colorado, "
            "covering November 2016 to October 2021, including the switch from free to paid charging and the COVID-19 period.")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig = px.histogram(C, x=TARGET, nbins=60, color_discrete_sequence=[ACCENT], labels={TARGET: "Delivered kWh"})
        chart(style(fig, "Delivered Energy Per Session").update_yaxes(title="Sessions"), "dist")
    with c2:
        s = C.sample(min(4000, len(C)), random_state=1)
        fig = px.scatter(s, x="kwh_requested", y=TARGET, opacity=0.35, color_discrete_sequence=[ACCENT],
                         labels={"kwh_requested": "Requested kWh", TARGET: "Delivered kWh"})
        fig.add_shape(type="line", x0=0, y0=0, x1=70, y1=70, line=dict(dash="dash", color=INK, width=1.5))
        fig.update_xaxes(range=[0, 70]); fig.update_yaxes(range=[0, 70])
        chart(style(fig, "Requested Vs Delivered Energy"), "scatter")
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        hourly = C.groupby("request_hour").size().reset_index(name="Sessions")
        fig = px.bar(hourly, x="request_hour", y="Sessions", color_discrete_sequence=[ACCENT], labels={"request_hour": "Hour Of Day"})
        chart(style(fig, "When Requests Arrive"), "hour")
    with c4:
        monthly = C.set_index("request_entry_time").resample("MS").size().reset_index(name="Sessions")
        fig = px.area(monthly, x="request_entry_time", y="Sessions", color_discrete_sequence=[ACCENT], labels={"request_entry_time": "Month"})
        chart(style(fig, "Sessions Per Month (Note The COVID-19 Drop)"), "month")

    section("How It Works:", "Simple 6 Steps To A Prediction.",
            "From raw charging records to a forecast an operator can act on.")
    steps = [("01", "Load The Data", "40,979 NREL charging requests with 16 columns are loaded and profiled."),
             ("02", "Clean The Records", "Duplicates, missing targets, invalid requests and 0 kWh sessions are removed."),
             ("03", "Engineer Features", "Request time, parking window and each driver's past behaviour become inputs."),
             ("04", "Split By Time", "Older sessions train the model; the newest 15% test it, just like real life."),
             ("05", "Train Three Models", "Linear Regression, Random Forest and XGBoost are trained and tuned."),
             ("06", "Evaluate & Explain", "MAE, RMSE and R² are compared against simple baselines and explained.")]
    a, mid, b = st.columns([1, 0.8, 1], gap="large", vertical_alignment="center")
    for col, items, cls in [(a, steps[0::2], "right"), (b, steps[1::2], "")]:
        with col:
            for n, h, d in items:
                html(f'<div class="step {cls}"><div class="n">{n}</div><div class="h">{h}</div><div class="d">{d}</div></div>')
    with mid:
        html(f'<div class="charger">{CHARGER_SVG}</div>')

    with st.expander("Data Cleaning Steps"):
        table(D["cleaning"])

    section("Got Questions?", "Everything You Need To Know.")
    faqs = [
        ("What Does ChargeWise Predict?",
         "The energy (kWh) a charging session will actually deliver, using only information known when the driver makes the request."),
        ("Why Not Just Trust What The Driver Requests?",
         f"Drivers over-request. In the test period, planning from requests would overestimate demand by {over_request:.0%}."),
        ("What Are Spare Hours?",
         "Parking time minus the charging time needed. A car parked 8.5 hours that needs 1.6 hours of charging has 6.9 spare hours, "
         "so its charging can be moved away from busy periods."),
        ("Why Does COVID-19 Matter?",
         "Our test period (2020 to 2021) falls in the pandemic, when workplace behaviour changed. On pre-COVID data the model is far more accurate."),
        ("Is This Data From UTP?",
         "No. The data comes from NREL in Colorado, USA. Insights for UTP are illustrative and would need local validation."),
    ]
    for q, a_ in faqs:
        with st.expander(q):
            st.write(a_)

# ================================================================ MODEL RESULTS
elif page == "Model Results":
    section("Model Results:", "Which Model Predicts Best?",
            "Trained on older sessions and tested on the newest 15% (January 2020 to October 2021). "
            "Lower MAE and RMSE are better; higher R² is better.")
    val_best = D["tuning"].groupby("Model")["Validation MAE"].min()
    cols = st.columns(3, gap="medium")
    for col, name in zip(cols, ["Linear Regression", "Random Forest", "XGBoost"]):
        row = R.set_index("Model").loc[name]
        hot = name == BEST
        with col:
            html(f"""
            <div class="plan {'hot' if hot else ''}">
              <div class="name">{name} {'<span class="badge">Chosen Model</span>' if hot else ''}</div>
              <div class="big">{row['MAE']:.2f}<small> kWh</small></div>
              <div class="sub">Test MAE (Average Error Per Session)</div>
              <ul><li>Test RMSE <b>{row['RMSE']:.2f} kWh</b></li>
                  <li>Test R² <b>{row['R2']:.2f}</b></li>
                  <li>Validation MAE <b>{val_best[name]:.2f} kWh</b></li></ul>
            </div>""")

    html('<div style="height:22px"></div>')
    r = R.copy()
    r["Type"] = np.where(r["Model"].str.startswith("Baseline"), "Baseline", "ML Model")
    r["Label"] = r["Model"].map(tc)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig = px.bar(r.sort_values("MAE"), x="MAE", y="Label", color="Type", orientation="h",
                     color_discrete_map={"Baseline": GREY, "ML Model": ACCENT}, labels={"Label": "", "MAE": "MAE (kWh)"})
        fig.update_yaxes(autorange="reversed")
        chart(style(fig, "Test MAE By Model", height=380, legend=True), "mae")
    with c2:
        fig = px.scatter(P.sample(min(4000, len(P)), random_state=1), x=TARGET, y=PRED, opacity=0.35,
                         color_discrete_sequence=[ACCENT], labels={TARGET: "Actual kWh", PRED: "Predicted kWh"})
        fig.add_shape(type="line", x0=0, y0=0, x1=60, y1=60, line=dict(dash="dash", color=INK, width=1.5))
        chart(style(fig, f"Actual Vs Predicted ({BEST})", height=380), "avp")

    section("COVID-19 Check:", "Did Behaviour Change The Score?",
            "The main test period falls during COVID-19. Repeating the experiment on pre-COVID data shows how much error comes from that shift.")
    cv = D["covid"]
    main, pre = cv.iloc[0], cv.iloc[1]
    metric_cards([("Main Test (COVID Period)", f"R² {main['R2']:.2f}", f"MAE {main['MAE']:.2f} kWh"),
                  ("Pre-COVID Test", f"R² {pre['R2']:.2f}", f"MAE {pre['MAE']:.2f} kWh"),
                  ("Pre-COVID Baseline", f"R² {cv.iloc[2]['R2']:.2f}", f"Driver Past Average, MAE {cv.iloc[2]['MAE']:.2f} kWh")])
    html('<div class="banner info">Before COVID-19 the model clearly beats the simple baseline. The drop in the main test comes from '
         'changed workplace behaviour, so a real system would need regular retraining.</div>')

    with st.expander("All Results Table"):
        table(R)
    with st.expander("Hyperparameter Tuning (Validation Set)"):
        table(D["tuning"])

# ================================================================ DEMAND DRIVERS
elif page == "Demand Drivers":
    section("What Drives Demand:", "The Factors Behind Every Charge.",
            "Permutation importance: shuffle one input and measure how much worse predictions get. "
            "It shows what helps prediction, not cause and effect.")
    imp = D["importance"].copy()
    imp["Feature"] = imp["Feature"].map(FEATURE_LABELS)
    col = imp.columns[1]
    fig = px.bar(imp, x=col, y="Feature", orientation="h", color_discrete_sequence=[ACCENT],
                 labels={col: "Increase In Error When Shuffled (kWh)", "Feature": ""})
    fig.update_yaxes(autorange="reversed")
    chart(style(fig, f"Feature Importance ({BEST})", height=460), "imp")

    p = P.copy()
    p["abs_error"] = (p[TARGET] - p[PRED]).abs()
    p["Demand Level"] = pd.cut(p[TARGET], [0, 5, 10, 20, 100],
                               labels=["Low<br>(<5 kWh)", "Medium<br>(5-10 kWh)", "High<br>(10-20 kWh)", "Very High<br>(>20 kWh)"])
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        e = p.groupby("Demand Level", observed=True)["abs_error"].mean().reset_index()
        fig = px.bar(e, x="Demand Level", y="abs_error", color_discrete_sequence=[ACCENT], labels={"abs_error": "MAE (kWh)"})
        chart(style(fig, "Error By Demand Level"), "lvl")
    with c2:
        e = p.groupby("request_hour")["abs_error"].mean().reset_index()
        fig = px.bar(e, x="request_hour", y="abs_error", color_discrete_sequence=[ACCENT_SOFT],
                     labels={"abs_error": "MAE (kWh)", "request_hour": "Request Hour"})
        chart(style(fig, "Error By Request Hour"), "hr")
    html('<div class="banner info">Errors are highest at the extremes. Very small sessions tend to be over-predicted and very large '
         'ones under-predicted, while typical 5 to 10 kWh sessions are predicted best.</div>')

# ================================================================ PREDICT
elif page == "Predict":
    section("Predict A Session:", "Enter A Request, Get A Forecast.",
            "Fill in what a driver provides when they request a charge.")
    new_driver = st.toggle("New Driver (No Charging History)", value=False)
    with st.form("predict"):
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            station = st.selectbox("Station", meta["stations"])
            OTHER = "Other / Not Listed"
            vms = [OTHER] + [v for v in meta["vehicle_models"] if v != "Unknown"]
            vehicle = st.selectbox("Vehicle Model (Optional)", vms, index=0,
                                   help="Leave as Other / Not Listed if the car is not in the list. "
                                        "It has very little effect on the prediction.")
            paid = st.radio("Charging Policy", ["Paid", "Free"], horizontal=True)
        with c2:
            km = st.number_input("Driving Distance Needed (km)", 1, 480, 65,
                                 help="How far the driver wants to be able to drive after charging. "
                                      "The energy needed (kWh) is calculated automatically from this and the car.")
            st.caption("Energy requested (kWh) is calculated automatically: km × the car's energy use per km "
                       f"(about {KWH_PER_MILE_DEFAULT / KM_PER_MILE:.2f} kWh per km for unlisted cars).")
            DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            req_day = st.selectbox("Request Day", DAYS, index=date.today().weekday())
        with c3:
            req_time = st.time_input("Request Time", time(8, 30))
            dep_time = st.time_input("Expected Departure", time(17, 0))
            power = st.selectbox("Charger Power (kW)", [3.3, 6.6, 7.2], index=1)
        st.markdown("**Driver History**" + (" (Not Used: New Driver)" if new_driver else ""))
        h1, h2, h3 = st.columns(3, gap="medium")
        past_n = h1.number_input("Past Sessions", 0, 2000, 30, disabled=new_driver)
        past_avg = h2.number_input("Past Average Delivered kWh", 0.0, 80.0, 9.0, disabled=new_driver)
        past_pct = h3.number_input("Usually Takes % Of Request", 0, 300, 70, disabled=new_driver)
        submitted = st.form_submit_button("Predict")

    if submitted:
        car = "Unknown" if vehicle == OTHER else vehicle
        rate = KWH_PER_MILE.get(car, KWH_PER_MILE_DEFAULT) if car != "Unknown" else KWH_PER_MILE_DEFAULT
        miles = km / KM_PER_MILE
        rate_km = round(rate / KM_PER_MILE, 3)
        kwh_req = round(km * rate_km, 2)
        # Build a date from the chosen weekday (this week); the exact date barely affects the model
        today = date.today()
        req_date = today + timedelta(days=DAYS.index(req_day) - today.weekday())
        start = datetime.combine(req_date, req_time)
        end = datetime.combine(req_date, dep_time)
        if end <= start:
            end += timedelta(days=1)
        parking = (end - start).total_seconds() / 3600
        fill = meta["fill_values"]
        row = {
            "kwh_requested": kwh_req, "miles_requested": miles, "parking_hours": parking,
            "request_hour": start.hour, "day_of_week": start.weekday(), "month": start.month,
            "is_weekend": int(start.weekday() >= 5), "paid_period": int(paid == "Paid"),
            "driver_past_sessions": 0 if new_driver else past_n,
            "driver_past_avg_kwh": fill["driver_past_avg_kwh"] if new_driver else past_avg,
            "driver_past_ratio": fill["driver_past_ratio"] if new_driver else past_pct / 100,
            "station": station, "vehicle_model": car,
        }
        X = pd.DataFrame([row])[meta["num_features"] + meta["cat_features"]]
        pred = max(float(model.predict(X)[0]), 0.0)
        hours = pred / power
        spare = parking - hours
        metric_cards([("kWh Requested (Auto)", f"{kwh_req:.1f} kWh", f"{km} km × {rate_km} kWh Per km"),
                      ("Predicted Energy", f"{pred:.1f} kWh", f"{pred - kwh_req:+.1f} kWh Vs Request"),
                      ("Charging Time Needed", f"{hours:.1f} h", f"At {power} kW"),
                      ("Parking Window", f"{parking:.1f} h", f"{start:%H:%M} To {end:%H:%M}"),
                      ("Spare (Flexible) Time", f"{max(spare, 0):.1f} h", "Time Available To Shift Charging")])
        if spare >= 1:
            html(f'<div class="banner ok"><b>Flexible Session.</b> About {spare:.1f} spare hours, so charging can be '
                 'delayed away from busy periods and still finish before departure.</div>')
        elif spare >= 0:
            html('<div class="banner warn"><b>Tight Session.</b> Little spare time. Start charging soon to finish before departure.</div>')
        else:
            html('<div class="banner bad"><b>At Risk.</b> The car may not reach the predicted energy before departure at this '
                 'charger power. Prioritise it or use a faster charger.</div>')
        st.caption("Charging time = predicted kWh ÷ charger power (simplified; ignores battery taper).")

# ================================================================ SMART CHARGING
elif page == "Smart Charging":
    section("Smart Charging:", "Shift The Load, Not The Driver.",
            "A simulation on test-period sessions with an assumed charger power. "
            "Results are illustrative, not measured reductions at NREL or UTP.")
    power = st.segmented_control("Assumed Charger Power (kW)", [3.3, 6.6, 7.2], default=6.6,
                                 format_func=lambda v: f"{v} kW") or 6.6
    s = P.copy()
    s["hours_needed"] = s[PRED] / power
    s["spare_hours"] = (s["parking_hours"] - s["hours_needed"]).clip(lower=0)
    stats([(f"{s['hours_needed'].mean():.1f} h", "Average Charging Time Needed"),
           (f"{s['parking_hours'].mean():.1f} h", "Average Parking Time"),
           (f"{(s['spare_hours'] >= 1).mean():.0%}", "Sessions With ≥1 Spare Hour")])

    totals = pd.DataFrame({
        "Planning Basis": ["Driver Requests", f"Model Prediction<br>({BEST})", "Actually Delivered"],
        "Total kWh": [s["kwh_requested"].sum(), s[PRED].sum(), s[TARGET].sum()],
    })
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig = px.bar(totals, x="Planning Basis", y="Total kWh", color="Planning Basis",
                     color_discrete_sequence=[GREY, ACCENT, INK], labels={"Planning Basis": ""})
        chart(style(fig, "Energy The Campus Would Plan For"), "plan")
    with c2:
        fig = px.histogram(s, x="spare_hours", nbins=40, color_discrete_sequence=[ACCENT],
                           labels={"spare_hours": "Spare Hours"})
        chart(style(fig, "Spare Hours Per Session").update_yaxes(title="Sessions"), "spare")

    model_over = totals["Total kWh"][1] / totals["Total kWh"][2] - 1
    html(f"""
    <div class="cta"><div class="t">Plan For Real Demand,<br>Not Requested Demand.</div>
    <div class="s">Planning from driver requests overestimates energy by <b>{over_request:.0%}</b>; the model's estimate is off by
    <b>{model_over:+.0%}</b>. Most cars park far longer than they need to charge, so charging can be spread across the day
    instead of all at once on arrival.</div></div>""")

footer()
