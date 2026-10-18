from pathlib import Path
import html
import math
import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

ROOT = Path(__file__).parent
DATA_PATHS = [
    ROOT / "predictive_output.csv",
    ROOT / "fused_tn_pumps.csv",
]
HOTSPOT_PATH = ROOT / "tamilnadu_hotspot_clusters.csv"

st.set_page_config(
    page_title="JalShield — Water, watched with care",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Design system ---------------------------
def css():
    st.markdown(r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Nunito:wght@500;600;700;800&display=swap');
    :root{
      --paper:#F8F5ED; --paper2:#F1EEE4; --card:#FFFCF5; --ink:#122C42;
      --muted:#66747A; --line:#E4DED0; --teal:#155C62; --teal2:#DCEBE5;
      --sage:#E8EEDC; --blue:#E3EDF1; --amber:#F8EED8; --coral:#FBE5DC;
      --green:#277B5B; --orange:#EA8555; --red:#D95345;
    }
    html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
    .stApp{background:var(--paper);color:var(--ink);}
    [data-testid="stHeader"]{background:rgba(248,245,237,.96);}
    [data-testid="stSidebar"]{background:#F1EEE4;border-right:1px solid #E3DDCF;}
    [data-testid="stSidebar"] > div{padding-top:1rem;}
    .block-container{max-width:1680px;padding-top:1.0rem;padding-bottom:2rem;}
    h1,h2,h3{font-family:'Nunito',sans-serif!important;color:var(--ink)!important;letter-spacing:-.035em;}
    h1{font-size:2.05rem!important;font-weight:800!important;line-height:1.12!important;}
    h2{font-size:1.25rem!important;font-weight:800!important;}
    h3{font-size:1rem!important;font-weight:800!important;}
    p, label, .stMarkdown{color:var(--ink);}
    [data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:17px;padding:14px 16px;box-shadow:0 3px 10px rgba(32,49,42,.025);}
    [data-testid="stMetricLabel"]{font-size:.82rem;color:var(--muted);}
    [data-testid="stMetricValue"]{font-family:'Nunito',sans-serif;font-weight:800;color:var(--ink);}
    [data-testid="stMetricDelta"]{font-size:.76rem;}
    .brand{display:flex;align-items:center;gap:10px;margin:2px 0 22px 2px;}
    .brand-mark{width:46px;height:46px;display:flex;align-items:center;justify-content:center;border-radius:17px;background:#DCEBE5;border:1px solid #C8DCD3;font-size:27px;}
    .brand-name{font:800 1.42rem 'Nunito',sans-serif;letter-spacing:-.04em;line-height:1;color:var(--ink);}
    .brand-sub{font-size:.57rem;letter-spacing:.12em;line-height:1.5;margin-top:4px;color:#5D7075;}
    .nav-caption{font-size:.68rem;letter-spacing:.12em;font-weight:700;color:#74817D;margin:10px 0 5px;}
    .eyebrow{font-size:.69rem;letter-spacing:.14em;font-weight:700;color:var(--teal);text-transform:uppercase;margin-bottom:5px;}
    .subtitle{font-size:.93rem;color:#5F6D72;margin-top:-6px;}
    .greeting{font:600 1.04rem 'Newsreader',serif;color:#315D56;letter-spacing:.01em;}
    .hero-strip{background:linear-gradient(105deg,#F8F5ED 0%,#F8F5ED 57%,#E7EFE4 100%);border:1px solid #E8E1D3;border-radius:20px;padding:18px 21px;position:relative;overflow:hidden;margin-bottom:16px;}
    .hero-note{font:italic 500 1.02rem 'Newsreader',serif;color:#42685F;}
    .hero-art{position:absolute;right:12px;bottom:-4px;opacity:.9;width:42%;height:92px;pointer-events:none;}
    .section-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:3px;}
    .section-title{font:800 1.12rem 'Nunito',sans-serif;color:var(--ink);}
    .section-sub{font-size:.79rem;color:var(--muted);margin-bottom:10px;}
    .panel{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px 17px;}
    .metric-label{font-size:.76rem;color:#52646B;}
    .metric-number{font:800 1.75rem 'Nunito',sans-serif;letter-spacing:-.04em;color:var(--ink);line-height:1.15;}
    .metric-foot{font-size:.7rem;color:#68767A;margin-top:4px;}
    .metric-emoji{font-size:1.6rem;line-height:1;}
    .risk-pill{display:inline-block;padding:4px 9px;border-radius:99px;font-size:.69rem;font-weight:700;}
    .risk-high{background:#FBE4D9;color:#B64E37;}
    .risk-medium{background:#F8EBC9;color:#8B6418;}
    .risk-low{background:#DDECE1;color:#286B4F;}
    .note-box{border-radius:14px;padding:11px 13px;background:#EDF3F4;border:1px solid #E0E9E8;font-size:.81rem;color:#345767;}
    .priority-item{display:grid;grid-template-columns:29px 1fr auto;gap:10px;align-items:center;padding:12px 0;border-bottom:1px solid #ECE7DC;}
    .priority-item:last-child{border-bottom:0;}
    .rank{width:28px;height:28px;display:flex;align-items:center;justify-content:center;border-radius:9px;font-weight:800;background:#FBE5DC;color:#B84D3D;}
    .district-name{font-weight:700;font-size:.86rem;color:var(--ink);}
    .district-action{font-size:.74rem;color:var(--muted);margin-top:3px;}
    .score{font-size:.76rem;font-weight:700;color:#B9573E;white-space:nowrap;}
    .tiny{font-size:.7rem;color:var(--muted);}
    .sensor-tile{background:#FFFCF6;border:1px solid #E9E3D8;border-radius:13px;padding:12px;min-height:105px;}
    .sensor-name{font-size:.73rem;color:#586B70;}
    .sensor-val{font:800 1.15rem 'Nunito',sans-serif;color:var(--ink);margin-top:7px;}
    .sensor-status{font-size:.68rem;color:#267653;margin-top:4px;}
    .insight{border-radius:14px;padding:12px 13px;margin-top:8px;background:#E9F0E7;font-size:.78rem;line-height:1.55;color:#34564D;}
    .insight.blue{background:#E9F0F2;color:#315667;}
    .footer-note{text-align:center;color:#8A938B;font-size:.7rem;padding:18px 0 3px;}
    .stButton>button{border-radius:12px;border:1px solid #DCD6C9;background:#FFFCF5;color:var(--ink);font-weight:600;min-height:2.35rem;}
    .stButton>button:hover{border-color:var(--teal);color:var(--teal);background:#F2F7F3;}
    div[data-testid="stSelectbox"] div[role="combobox"],div[data-testid="stTextInput"] input{border-radius:12px!important;background:#FFFCF5!important;}
    div[data-testid="stRadio"] label{font-size:.88rem;}
    [data-testid="stSidebar"] [data-testid="stRadio"] > label{font-size:.7rem!important;letter-spacing:.1em!important;font-weight:700!important;color:#74817D!important;}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:3px;}
    [data-testid="stSidebar"] .stRadio label{padding:7px 9px;border-radius:10px;}
    [data-testid="stSidebar"] .stRadio label:has(input:checked){background:#DCE6D8;}
    iframe{border-radius:14px;}
    footer{visibility:hidden;}

    /* JalShield visual polish pass */
    .block-container{max-width:1760px;padding-top:.72rem;padding-bottom:1.35rem;}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#F1EEE4 0%,#F5F1E8 100%);}
    [data-testid="stSidebar"] .brand{padding:5px 4px 15px;border-bottom:1px solid #E0D9CB;}
    .brand-mark{box-shadow:0 3px 0 rgba(21,92,98,.06);font-size:25px;}
    .brand-name{font-size:1.5rem;}
    [data-testid="stSidebar"] [data-testid="stRadio"] > label{margin-bottom:8px;}
    [data-testid="stSidebar"] div[role="radiogroup"]{gap:5px!important;}
    [data-testid="stSidebar"] div[role="radiogroup"] label{
      display:flex;align-items:center;gap:8px;padding:9px 11px!important;
      border-radius:12px!important;border:1px solid transparent;transition:all .15s ease;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:#E8E9DE;border-color:#DCE0D3;}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
      background:#DCE7D9!important;border:1px solid #CFDCCB!important;
      box-shadow:inset 3px 0 #2C756B;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child{display:none!important;}
    [data-testid="stSidebar"] div[role="radiogroup"] label p{font-weight:600!important;font-size:.87rem!important;}
    .eyebrow{font-size:.65rem;letter-spacing:.17em;color:#6E8C7E;}
    h1{font-size:2.22rem!important;letter-spacing:-.045em!important;}
    .subtitle{font-size:.91rem;max-width:760px;}
    .hero-strip{padding:17px 21px 16px;margin-bottom:13px;min-height:102px;}
    .hero-strip .greeting{font-size:1rem;}
    .hero-strip .hero-art{height:100px;width:40%;opacity:.83;}
    [data-testid="stMetric"]{
      min-height:118px;padding:13px 15px 11px!important;border-radius:16px!important;
      background:rgba(255,252,245,.88);box-shadow:0 4px 13px rgba(35,55,42,.035);
      display:flex;flex-direction:column;justify-content:center;
    }
    [data-testid="stMetricLabel"]{font-size:.78rem!important;font-weight:600!important;color:#5D7075!important;}
    [data-testid="stMetricValue"]{font-size:1.92rem!important;line-height:1.05!important;margin-top:5px;}
    [data-testid="stMetricDelta"]{font-size:.72rem!important;}
    [data-testid="stCaptionContainer"] p{font-size:.7rem;color:#7C8986;}
    .panel{box-shadow:0 4px 14px rgba(35,55,42,.025);}
    .section-title{font-size:1.12rem;letter-spacing:-.025em;}
    .section-sub{font-size:.76rem;margin-bottom:9px;}
    div[data-testid="stSelectbox"] div[role="combobox"]{border-radius:12px!important;min-height:42px;}
    div[data-testid="stSegmentedControl"]{background:#EAE6DC;border:1px solid #DED7C9;border-radius:999px;padding:3px;}
    div[data-testid="stSegmentedControl"] button{border-radius:999px!important;font-size:.79rem!important;}
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"]{background:#155C62!important;color:#fff!important;}
    [data-testid="stAlert"]{border-radius:14px!important;border:1px solid #DDE7E5!important;}
    .stDataFrame{border:1px solid #E4DED0;border-radius:12px;overflow:hidden;}
    .stDownloadButton button{border-radius:12px;}
    @media(max-width:1100px){
      h1{font-size:1.75rem!important;}
      [data-testid="stMetricValue"]{font-size:1.55rem!important;}
      .hero-strip .hero-art{opacity:.25;width:55%;}
    }
    @media(max-width:700px){
      .block-container{padding-left:.7rem;padding-right:.7rem;}
      h1{font-size:1.5rem!important;}
      .hero-strip .hero-art{display:none;}
      [data-testid="stMetric"]{min-height:100px;}
    }

    @media(max-width:1100px){
      .block-container{padding-left:1rem;padding-right:1rem;}
      h1{font-size:1.65rem!important;}
      .hero-art{opacity:.35;width:52%;}
    }
    @media(max-width:700px){
      .block-container{padding-left:.65rem;padding-right:.65rem;}
      .hero-art{display:none;}
    }
    </style>
    """, unsafe_allow_html=True)

css()

# --------------------------- Data loading ---------------------------
@st.cache_data(show_spinner=False)
def load_data():
    for path in DATA_PATHS:
        if path.exists():
            try:
                frame = pd.read_csv(path, low_memory=False)
                if not frame.empty:
                    return frame, path.name
            except Exception:
                pass
    return pd.DataFrame(), None

@st.cache_data(show_spinner=False)
def load_hotspots():
    if HOTSPOT_PATH.exists():
        try:
            return pd.read_csv(HOTSPOT_PATH, low_memory=False)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

df, data_source = load_data()
hotspots = load_hotspots()
demo_mode = df.empty

def numeric(col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def first_col(frame, names):
    return next((c for c in names if c in frame.columns), None)

def safe_text(value, fallback="Not available"):
    if pd.isna(value):
        return fallback
    return str(value)

def risk_colour(category):
    c = str(category).lower()
    if "high" in c or "critical" in c or "broken" in c:
        return "#D95345"
    if "medium" in c or "recover" in c or "moderate" in c:
        return "#E9A14A"
    if "low" in c or "normal" in c:
        return "#4B9472"
    return "#7E9995"

def map_frame(mode):
    if df.empty:
        return pd.DataFrame([
            {"district":"Coimbatore","latitude":11.0168,"longitude":76.9558,"Priority_Score":82,"Intervention_Category":"High"},
            {"district":"Ramanathapuram","latitude":9.3639,"longitude":78.8395,"Priority_Score":78,"Intervention_Category":"High"},
            {"district":"Thoothukudi","latitude":8.7642,"longitude":78.1348,"Priority_Score":71,"Intervention_Category":"Medium"},
            {"district":"Tirunelveli","latitude":8.7139,"longitude":77.7567,"Priority_Score":68,"Intervention_Category":"Medium"},
            {"district":"Chennai","latitude":13.0827,"longitude":80.2707,"Priority_Score":44,"Intervention_Category":"Low"},
        ])
    frame = df.copy()
    lat = first_col(frame, ["latitude", "Latitude", "lat"])
    lon = first_col(frame, ["longitude", "Longitude", "lon", "lng"])
    if not lat or not lon:
        return pd.DataFrame()
    frame[lat] = pd.to_numeric(frame[lat], errors="coerce")
    frame[lon] = pd.to_numeric(frame[lon], errors="coerce")
    frame = frame.dropna(subset=[lat, lon])
    frame = frame[frame[lat].between(7.2, 14.2) & frame[lon].between(75.5, 81.8)]
    if frame.empty:
        return frame
    frame = frame.rename(columns={lat:"latitude", lon:"longitude"})
    if mode == "Projected · 2031" and "Priority_Score_2031" in frame:
        frame["map_score"] = pd.to_numeric(frame["Priority_Score_2031"], errors="coerce")
        cat = "Intervention_Category_2031_fixed_cuts"
    else:
        frame["map_score"] = pd.to_numeric(frame.get("Priority_Score", pd.Series(index=frame.index,dtype=float)), errors="coerce")
        cat = "Intervention_Category"
    frame["map_category"] = frame[cat].astype(str) if cat in frame.columns else "Unrated"
    if mode == "Groundwater level" and "groundwater_level_m" in frame:
        frame["map_category"] = pd.to_numeric(frame["groundwater_level_m"], errors="coerce").apply(
            lambda x: "High" if pd.notna(x) and x >= 30 else ("Medium" if pd.notna(x) and x >= 15 else "Low"))
    return frame

def make_map(frame, layer, full=False):
    fmap = folium.Map(location=[10.8,78.3], zoom_start=7 if full else 6, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="Street atlas", control=False, show=True).add_to(fmap)
    if frame.empty:
        folium.Marker([10.8,78.3], tooltip="Tamil Nadu", popup="Add coordinates to the dataset to plot locations.").add_to(fmap)
        return fmap
    # Reduce visual clutter on overview, but retain all locations on expanded map.
    plot = frame if full else frame.head(650)
    if layer == "Priority heat":
        pts = [[r.latitude, r.longitude, max(0.01, float(r.map_score) if pd.notna(r.map_score) else 0.1)] for r in plot.itertuples()]
        HeatMap(pts, radius=18, blur=20, min_opacity=.25, max_zoom=12).add_to(fmap)
    else:
        cluster = MarkerCluster(name="Mapped reference locations", show=True).add_to(fmap) if full else None
        for r in plot.itertuples():
            color = risk_colour(getattr(r, "map_category", "Unrated"))
            district = html.escape(str(getattr(r, "district", "Location")))
            score = getattr(r, "map_score", float("nan"))
            popup = f"<div style='font-family:Arial;min-width:170px'><b>{district}</b><br>Category: {html.escape(str(getattr(r,'map_category','Unrated')))}"
            if pd.notna(score):
                popup += f"<br>Priority screening score: {float(score):.2f}"
            if "geospatial_data_quality" in frame.columns:
                quality = html.escape(str(getattr(r, "geospatial_data_quality", "unknown")))
                popup += f"<br><small>Location quality: {quality}</small>"
            popup += "</div>"
            marker = folium.CircleMarker(
                location=[float(r.latitude), float(r.longitude)], radius=5 if not full else 4,
                color="#FFFCF5", weight=1, fill=True, fill_color=color, fill_opacity=.86,
                tooltip=district, popup=folium.Popup(popup, max_width=260),
            )
            marker.add_to(cluster if cluster is not None else fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-mark">💧</div>
      <div><div class="brand-name">JalShield</div><div class="brand-sub">WATER INFRASTRUCTURE<br>INTELLIGENCE</div></div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio(
        "YOUR WORKSPACE",
        ["Overview", "Predictions", "Water Quality", "Maintenance & Dispatch", "Live Sensors", "Data Explorer"],
        index=0,
        label_visibility="visible",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="hero-note">A little more care,<br>a little less crisis.</div>', unsafe_allow_html=True)
    st.markdown('<div class="tiny" style="margin-top:8px">Tamil Nadu · Prototype</div>', unsafe_allow_html=True)

# --------------------------- Header ---------------------------
top1, top2 = st.columns([2.3, 1.15], vertical_alignment="center")
with top1:
    st.markdown('<div class="eyebrow">WATER, WATCHED WITH CARE</div>', unsafe_allow_html=True)
    if page == "Overview":
        st.title("A clearer view of Tamil Nadu.")
        st.markdown('<div class="subtitle">A calm overview of water infrastructure, groundwater and the next best action.</div>', unsafe_allow_html=True)
    else:
        titles = {
            "Predictions": ("Looking ahead, thoughtfully.", "Experimental scenarios, clearly separated from observed data."),
            "Water Quality": ("Water quality, district by district.", "Review available chemistry indicators and their data coverage."),
            "Maintenance & Dispatch": ("Maintenance, planned with purpose.", "A budget- and crew-aware dispatch plan for human review."),
            "Live Sensors": ("Listening to the infrastructure.", "Hardware readings appear here when an ESP32 feed is connected."),
            "Data Explorer": ("Get closer to the data.", "Explore the source fields, coverage and missing values."),
        }
        title, subtitle = titles.get(page, ("JalShield", "Water infrastructure intelligence"))
        st.title(title)
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
with top2:
    view = st.segmented_control("VIEW", ["Current · 2026", "Projected · 2031"], default="Current · 2026", key="scenario_mode")
    st.markdown('<div style="text-align:right;font-size:.7rem;color:#718079;margin-top:2px">Prototype · not an operational alert system</div>', unsafe_allow_html=True)

st.markdown("")

# Demo/data notice
if demo_mode:
    st.info("Preview mode: no predictive_output.csv or fused_tn_pumps.csv was found. The map and priority list use illustrative examples.", icon="ℹ️")
else:
    st.markdown(
        f'<div class="note-box">Showing <b>{len(df):,}</b> records from <b>{html.escape(data_source or "dataset")}</b>. '
        'Outputs are experimental screening results; reference coordinates are not necessarily verified pump locations.</div>',
        unsafe_allow_html=True
    )

# --------------------------- Overview ---------------------------
if page == "Overview":
    # metrics based on available data when possible
    if not demo_mode:
        cat_col = first_col(df, ["Intervention_Category"])
        score_col = first_col(df, ["Priority_Score"])
        crit = int(df[cat_col].astype(str).str.lower().isin(["high","critical"]).sum()) if cat_col else 0
        pop_col = first_col(df, ["population_served"])
        pop = pd.to_numeric(df.loc[df[cat_col].astype(str).str.lower().isin(["high","critical"]), pop_col], errors="coerce").sum() if (cat_col and pop_col) else float("nan")
        crew_text = "Not connected"
        cvi_col = first_col(df, ["Climate_Vulnerability_Index"])
        cvi = pd.to_numeric(df[cvi_col], errors="coerce").mean() if cvi_col else float("nan")
        critical_text = f"{crit:,}"
        pop_text = f"{pop/1000:.1f}k" if pd.notna(pop) else "—"
        cvi_text = f"{cvi:.0f} / 100" if pd.notna(cvi) and cvi <= 100 else (f"{cvi:.2f}" if pd.notna(cvi) else "—")
    else:
        crit, critical_text, pop_text, crew_text, cvi_text = 12, "12", "24.6k", "8 / 10", "76 / 100"

    # Illustrated but restrained hero banner, built with inline vector art
    st.markdown(r"""
    <div class="hero-strip">
      <div class="greeting">Good morning. <span style="color:#D99028">☀</span></div>
      <div style="font:500 1.18rem 'Newsreader',serif;color:#19344A;margin-top:3px">Water is a shared resource. Let's look after it.</div>
      <div class="hero-note" style="margin-top:5px">Healthy aquifers · stronger tomorrows</div>
      <svg class="hero-art" viewBox="0 0 500 110" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
        <path d="M0 91 Q45 54 85 81 T175 70 T255 78 T340 62 T500 81 L500 110 L0 110Z" fill="#D8E6D3"/>
        <path d="M115 91 L174 22 L216 76 L247 46 L294 90Z" fill="#A9C6B5"/>
        <path d="M165 35 L174 22 L190 46 L179 42 L173 48Z" fill="#F8F5ED"/>
        <path d="M245 90 Q280 58 315 88 T380 83 T450 91" fill="none" stroke="#7DAFA7" stroke-width="3" stroke-linecap="round"/>
        <path d="M45 94 Q72 78 94 94" fill="none" stroke="#C3A66A" stroke-width="2"/>
        <g transform="translate(345 32)" stroke="#3C7E69" stroke-width="2" fill="none" stroke-linecap="round">
          <path d="M0 60 V8"/><path d="M0 19 Q-22 9 -20 0 Q-3 3 0 19"/><path d="M0 26 Q20 12 24 4 Q7 6 0 26"/>
          <path d="M0 37 Q-20 25 -25 17 Q-7 19 0 37"/><path d="M0 44 Q20 29 29 23 Q9 28 0 44"/>
        </g>
        <g transform="translate(420 45)" stroke="#3C7E69" stroke-width="2" fill="none" stroke-linecap="round">
          <path d="M0 48 V6"/><path d="M0 17 Q-16 9 -17 0 Q-3 3 0 17"/><path d="M0 23 Q16 11 21 4 Q6 6 0 23"/>
        </g>
        <path d="M0 99 Q100 95 200 101 T500 98" fill="none" stroke="#C9B477" stroke-width="2"/>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4, gap="small")
    with m1:
        st.metric("🚨  High-priority locations", critical_text)
        st.caption("Screening queue · review needed")
    with m2:
        st.metric("👥  Population potentially affected", pop_text)
        st.caption("Estimated exposure · not verified")
    with m3:
        st.metric("🦺  Maintenance crews", crew_text)
        st.caption("Resource allocation module not connected")
    with m4:
        st.metric("🌱  Climate vulnerability", cvi_text)
        st.caption("Experimental screening indicator")

    st.markdown("")
    # map + priorities
    map_col, priority_col = st.columns([1.8, 1.0], gap="medium")
    with map_col:
        st.markdown('<div class="section-head"><div class="section-title">Tamil Nadu <span style="color:#6F9B8B">↗</span></div><span class="tiny">District-level screening view</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Explore mapped reference locations and risk layers.</div>', unsafe_allow_html=True)
        layer_col, expand_col = st.columns([1.4, .8], vertical_alignment="center")
        with layer_col:
            layer = st.selectbox("Map layer", ["Risk markers", "Priority heat", "Groundwater level"], label_visibility="collapsed", key="overview_layer")
        with expand_col:
            expand = st.toggle("Expand map", value=False, key="expand_map_toggle")
        frame = map_frame(view if view else "Current · 2026")
        if layer == "Groundwater level":
            frame = map_frame("Groundwater level")
        fmap = make_map(frame, layer, full=expand)
        st_folium(fmap, height=650 if expand else 360, use_container_width=True, key="main_map", returned_objects=[])
        st.markdown('<div class="tiny">Red = high · amber = medium · green = low. Locations may be proxies; see data quality.</div>', unsafe_allow_html=True)

    with priority_col:
        st.markdown('<div class="section-head"><div class="section-title">Worth a closer look</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">A short list to guide human review.</div>', unsafe_allow_html=True)
        if not demo_mode and "Priority_Score" in df.columns:
            q = df.copy()
            q["Priority_Score"] = pd.to_numeric(q["Priority_Score"], errors="coerce")
            q = q.dropna(subset=["Priority_Score"]).sort_values("Priority_Score", ascending=False)
            q = q.drop_duplicates(subset=["district"], keep="first") if "district" in q.columns else q.head(4)
            rows = q.head(5).to_dict("records")
        else:
            rows = [
                {"district":"Coimbatore","Priority_Score":82,"Intervention_Category":"High","Priority_note":"Inspect high-priority locations"},
                {"district":"Ramanathapuram","Priority_Score":78,"Intervention_Category":"High","Priority_note":"Review groundwater stress"},
                {"district":"Thoothukudi","Priority_Score":71,"Intervention_Category":"Medium","Priority_note":"Verify water-quality inputs"},
                {"district":"Tirunelveli","Priority_Score":68,"Intervention_Category":"Medium","Priority_note":"Review equipment indicators"},
            ]
        for i, row in enumerate(rows, start=1):
            district = safe_text(row.get("district", "Unknown district"), "Unknown district")
            score = pd.to_numeric(row.get("Priority_Score"), errors="coerce")
            cat = safe_text(row.get("Intervention_Category", "Unrated"))
            action = safe_text(row.get("Priority_note", ""), "")
            if not action or action == "Not available":
                action = "Review screening result and source data"
            badge_class = "risk-high" if "high" in cat.lower() else ("risk-medium" if "medium" in cat.lower() else "risk-low")
            score_text = f"{score:.2f}" if pd.notna(score) else "—"
            st.markdown(f"""
            <div class="priority-item">
              <div class="rank" style="background:{'#FBE5DC' if i==1 else ('#F8EBC9' if i in [2,3] else '#E1EBE6')};color:{'#B84D3D' if i==1 else ('#8B6418' if i in [2,3] else '#286B4F')}">{i}</div>
              <div><div class="district-name">{html.escape(str(district))}</div><div class="district-action">{html.escape(str(action)[:75])}</div></div>
              <div style="text-align:right"><div class="score">PIS {score_text}</div><span class="risk-pill {badge_class}">{html.escape(cat)}</span></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="note-box"><b>Field note</b><br>Priority scores are screening indices. They help identify locations for review; they are not verified failure probabilities.</div>', unsafe_allow_html=True)

    st.markdown("")
    # bottom row
    sens_col, district_col, insight_col = st.columns([1.15,1,1], gap="medium")
    with sens_col:
        st.markdown('<div class="section-title">Live sensor feed <span style="font-size:.75rem;color:#74817D;font-weight:500">(ESP32)</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Waiting for the hardware connection.</div>', unsafe_allow_html=True)
        s1,s2,s3 = st.columns(3, gap="small")
        for col, name, value, unit, symbol in [
            (s1,"Vibration","—","g","〰"),(s2,"Motor temp","—","°C","♨"),(s3,"TDS","—","ppm","〜")
        ]:
            with col:
                st.markdown(f'<div class="sensor-tile"><div class="sensor-name">{symbol} {name}</div><div class="sensor-val">{value} <span style="font-size:.7rem">{unit}</span></div><div class="sensor-status">○ Not connected</div></div>', unsafe_allow_html=True)
        st.caption("No values are presented as live until a device feed is connected.")
    with district_col:
        st.markdown('<div class="section-title">District comparison</div>', unsafe_allow_html=True)
        measure = st.selectbox("Compare by", ["Groundwater level", "JJM tap coverage", "Priority score"], label_visibility="collapsed", key="district_measure")
        if not demo_mode and "district" in df.columns:
            metric_map = {
                "Groundwater level": "groundwater_level_m",
                "JJM tap coverage": "jjm_tap_coverage_pct",
                "Priority score": "Priority_Score",
            }
            col = metric_map[measure]
            if col in df.columns:
                tmp = df[["district", col]].copy()
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
                comp = tmp.dropna().groupby("district")[col].median().sort_values(ascending=False).head(5)
                if not comp.empty:
                    comp_df = comp.rename("Value").reset_index()
                    st.dataframe(comp_df, hide_index=True, use_container_width=True, height=190)
                else:
                    st.caption("No usable values for this comparison yet.")
            else:
                st.caption("This field is not present in the current dataset.")
        else:
            st.caption("District comparisons will populate from source-backed groundwater and JJM fields.")
            st.markdown('<div class="tiny">Add the fused/predictive CSV to enable comparisons.</div>', unsafe_allow_html=True)
    with insight_col:
        st.markdown('<div class="section-title">Small observations</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight blue">🫧 <b>Check the source first.</b><br>Groundwater observations can differ by station, date and distance from the mapped location.</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight">🌿 <b>Plan for resilience.</b><br>Use the scenario view to explore possible future pressure, not as a guaranteed forecast.</div>', unsafe_allow_html=True)

# --------------------------- Other pages ---------------------------
elif page == "Predictions":
    st.markdown("### Predictive Scenarios & 2031 RUL Projections")
    st.markdown(
        "Explore how groundwater-extraction stress could change infrastructure priority by 2031. "
        "These are experimental scenarios, not validated forecasts."
    )

    stress_mult = st.slider(
        "Groundwater extraction stress multiplier (2026–2031)",
        min_value=1.0, max_value=2.0, value=1.25, step=0.05,
        help="A scenario control for exploring higher stress; it does not retrain or validate the prediction model.",
        key="prediction_stress_multiplier",
    )

    if not df.empty:
        priority_col = first_col(df, ["Priority_Score", "Priority score", "priority_score"])
        future_priority_col = first_col(df, ["Priority_Score_2031", "2031_Priority_Score", "Priority_Score_2031_pred"])
        district_col = first_col(df, ["district", "District"])
        rul_cols = [c for c in ["Aquifer_RUL_Months", "Fluoride_RUL_Months", "TDS_RUL_Months", "RUL_Months"] if c in df.columns]
        cvi_col = first_col(df, ["CVI", "Climate_Vulnerability_Index"])
        category_col = first_col(df, ["Intervention_Category_2031_fixed_cuts", "Intervention_Category_2031", "Intervention_Category", "Risk_Level"])

        scenario = df.copy()
        if priority_col:
            scenario[priority_col] = pd.to_numeric(scenario[priority_col], errors="coerce")
            scenario["Scenario priority (2031)"] = scenario[priority_col] * stress_mult
        if future_priority_col:
            scenario[future_priority_col] = pd.to_numeric(scenario[future_priority_col], errors="coerce")
            scenario["Modelled 2031 priority"] = scenario[future_priority_col]
            if stress_mult != 1.0 and priority_col:
                scenario["Scenario priority (2031)"] = scenario[future_priority_col] * stress_mult

        base_priority = pd.to_numeric(df[priority_col], errors="coerce") if priority_col else pd.Series(dtype=float)
        scenario_priority = scenario["Scenario priority (2031)"] if "Scenario priority (2031)" in scenario else pd.Series(dtype=float)
        projected_high = int((scenario_priority >= scenario_priority.quantile(0.75)).sum()) if not scenario_priority.dropna().empty else 0
        current_high = int((base_priority >= base_priority.quantile(0.75)).sum()) if not base_priority.dropna().empty else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Assets in dataset", f"{len(df):,}")
        m2.metric("High-priority in scenario", f"{projected_high:,}" if not scenario_priority.dropna().empty else "—")
        m3.metric("Stress multiplier", f"{stress_mult:.2f}×", delta=f"{(stress_mult - 1) * 100:.0f}% scenario stress")

        st.markdown("#### Regional 2031 Stress Projection Table")
        display_cols = []
        if district_col: display_cols.append(district_col)
        if priority_col: display_cols.append(priority_col)
        if future_priority_col and future_priority_col != priority_col: display_cols.append(future_priority_col)
        if "Scenario priority (2031)" in scenario.columns: display_cols.append("Scenario priority (2031)")
        display_cols.extend([c for c in rul_cols if c not in display_cols])
        if cvi_col and cvi_col not in display_cols: display_cols.append(cvi_col)
        if category_col and category_col not in display_cols: display_cols.append(category_col)

        if display_cols:
            table = scenario[display_cols].copy()
            table = table.rename(columns={
                priority_col: "Current priority score" if priority_col else "Priority score",
                future_priority_col: "Modelled 2031 priority" if future_priority_col else "2031 priority",
            })
            sort_target = "Scenario priority (2031)" if "Scenario priority (2031)" in table.columns else (
                "Modelled 2031 priority" if "Modelled 2031 priority" in table.columns else None
            )
            if sort_target:
                table = table.sort_values(sort_target, ascending=False, na_position="last")
            st.dataframe(table.head(100), use_container_width=True, hide_index=True, height=420)
            st.download_button(
                "Download 2031 scenario table (CSV)",
                table.to_csv(index=False).encode("utf-8"),
                "jalshield_2031_scenario.csv",
                "text/csv",
                key="download_prediction_scenario",
            )
        else:
            st.info("The loaded dataset does not contain recognized prediction columns yet.")

        if priority_col and future_priority_col:
            chart_data = pd.DataFrame({
                "Current priority": pd.to_numeric(df[priority_col], errors="coerce"),
                "Modelled 2031 priority": pd.to_numeric(df[future_priority_col], errors="coerce"),
            }).dropna().head(100)
            if not chart_data.empty:
                st.markdown("#### Current vs modelled priority")
                st.line_chart(chart_data.reset_index(drop=True), use_container_width=True)

        st.caption("Scenario scores are simple stress-adjusted comparisons. They should guide exploration, not be interpreted as calibrated probabilities or confirmed future failures.")
    else:
        st.info("Connect predictive_output.csv or fused_tn_pumps.csv to show scenario projections.")

elif page == "Water Quality":
    st.markdown("### Water Quality & Contamination Analytics")
    st.caption("District-level chemistry indicators from the loaded dataset. Synthetic or uncalibrated values are not verified water-test results.")

    if not df.empty:
        district_col = first_col(df, ["district", "District"])
        tds_col = first_col(df, ["TDS", "tds", "TDS_ppm"])
        fluoride_col = first_col(df, ["Fluoride_Level", "Fluoride", "fluoride"])
        nitrate_col = first_col(df, ["Nitrate_Level", "Nitrate", "nitrate"])
        wqi_col = first_col(df, ["WQI", "WQI_Score", "water_quality_index"])
        contam_col = first_col(df, ["Water_Contamination_Risk", "contamination_risk"])
        synth_col = first_col(df, ["chemistry_values_synthetic"])

        def numeric_series(col):
            return pd.to_numeric(df[col], errors="coerce") if col and col in df.columns else pd.Series(dtype=float)

        q1, q2, q3, q4 = st.columns(4)
        tds_values = numeric_series(tds_col)
        fluoride_values = numeric_series(fluoride_col)
        nitrate_values = numeric_series(nitrate_col)
        wqi_values = numeric_series(wqi_col)
        q1.metric("Average TDS", f"{tds_values.mean():.1f} ppm" if not tds_values.dropna().empty else "—")
        q2.metric("Average Fluoride", f"{fluoride_values.mean():.2f} mg/L" if not fluoride_values.dropna().empty else "—")
        q3.metric("Average Nitrate", f"{nitrate_values.mean():.1f} mg/L" if not nitrate_values.dropna().empty else "—")
        q4.metric("Average WQI", f"{wqi_values.mean():.1f}" if not wqi_values.dropna().empty else "—")

        if synth_col and df[synth_col].astype(str).str.lower().isin(["true", "1", "yes"]).any():
            st.warning("This dataset flags some chemistry values as synthetic. Treat these figures as illustrative, not measured contamination.")

        st.markdown("#### District Water Quality Breakdown")
        table = pd.DataFrame()
        table["District"] = df[district_col].astype(str) if district_col else "Unknown"
        if tds_col: table["TDS (ppm)"] = numeric_series(tds_col)
        if fluoride_col: table["Fluoride (mg/L)"] = numeric_series(fluoride_col)
        if nitrate_col: table["Nitrate (mg/L)"] = numeric_series(nitrate_col)
        if wqi_col: table["WQI"] = numeric_series(wqi_col)
        if contam_col: table["Contamination risk"] = numeric_series(contam_col)
        pop_col = first_col(df, ["population_served", "Population"])
        risk_col = first_col(df, ["Intervention_Category", "Risk_Level"])
        if pop_col: table["Population served"] = pd.to_numeric(df[pop_col], errors="coerce")
        if risk_col: table["Priority category"] = df[risk_col].astype(str)
        if synth_col: table["Chemistry synthetic?"] = df[synth_col].astype(str)

        filter_col1, filter_col2 = st.columns([1.3, 1])
        with filter_col1:
            district_options = ["All districts"] + sorted(table["District"].dropna().unique().tolist())
            selected_district = st.selectbox("Filter district", district_options, key="wq_district_filter")
        with filter_col2:
            sort_options = [c for c in ["TDS (ppm)", "Fluoride (mg/L)", "Nitrate (mg/L)", "WQI", "Contamination risk"] if c in table.columns]
            sort_by = st.selectbox("Sort table by", sort_options, key="wq_sort") if sort_options else None

        shown = table.copy()
        if selected_district != "All districts":
            shown = shown[shown["District"] == selected_district]
        if sort_by:
            shown = shown.sort_values(sort_by, ascending=False, na_position="last")
        st.dataframe(shown, use_container_width=True, hide_index=True, height=430)
        st.download_button("Download water quality table (CSV)", shown.to_csv(index=False).encode("utf-8"), "jalshield_water_quality.csv", "text/csv")
    else:
        st.info("Add predictive_output.csv or fused_tn_pumps.csv to populate water-quality indicators.")

elif page == "Maintenance & Dispatch":
    st.markdown("### Maintenance & Dispatch — Resource Allocation")
    st.caption("A decision-support plan using priority, estimated repair cost, available budget and crew capacity. This is not an automatic dispatch order.")

    if not df.empty:
        district_col = first_col(df, ["district", "District"])
        score_col = first_col(df, ["Priority_Score"])
        category_col = first_col(df, ["Intervention_Category", "Risk_Level"])
        population_col = first_col(df, ["population_served", "Population"])
        rul_col = first_col(df, ["Aquifer_RUL_Months", "RUL_Months"])
        if score_col:
            work = pd.DataFrame()
            work["District"] = df[district_col].astype(str) if district_col else "Unknown"
            work["Priority score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
            work["Population served"] = pd.to_numeric(df[population_col], errors="coerce").fillna(0) if population_col else 0
            work["Aquifer RUL (months)"] = pd.to_numeric(df[rul_col], errors="coerce") if rul_col else float("nan")
            work["Risk category"] = df[category_col].astype(str) if category_col else "Unrated"
            if "geospatial_data_quality" in df.columns:
                work["Location quality"] = df["geospatial_data_quality"].astype(str)

            st.markdown("#### Dispatch constraints")
            c1, c2 = st.columns(2)
            with c1:
                budget = st.slider("Available maintenance budget (₹)", 100000, 10000000, 2500000, 100000, key="dispatch_budget")
            with c2:
                crews = st.slider("Available field crews", 1, 30, 12, 1, key="dispatch_crews")

            # Retain the teammate's greedy cost/efficiency selection logic.
            work["Estimated repair cost (₹)"] = work["Priority score"] * 120000 + work["Population served"] * 3
            work["Efficiency"] = (work["Population served"] * work["Priority score"]) / work["Estimated repair cost (₹)"].replace(0, np.nan)
            work = work.sort_values("Efficiency", ascending=False, na_position="last")

            selected_rows = []
            spent = 0.0
            for _, row in work.iterrows():
                cost = float(row["Estimated repair cost (₹)"])
                if len(selected_rows) < crews and spent + cost <= budget:
                    selected_rows.append(row)
                    spent += cost
            selected = pd.DataFrame(selected_rows)

            m1, m2, m3 = st.columns(3)
            m1.metric("Assets selected", f"{len(selected)}")
            m2.metric("Budget allocated", f"₹{spent:,.0f}")
            m3.metric("Budget remaining", f"₹{budget-spent:,.0f}")

            st.markdown("#### Optimal Procurement & Dispatch Manifest")
            if not selected.empty:
                selected = selected.sort_values("Priority score", ascending=False)
                st.dataframe(selected.drop(columns=["Efficiency"], errors="ignore"), use_container_width=True, hide_index=True, height=420)
                st.download_button("Download dispatch manifest (CSV)", selected.drop(columns=["Efficiency"], errors="ignore").to_csv(index=False).encode("utf-8"), "jalshield_dispatch_manifest.csv", "text/csv")
            else:
                st.warning("No assets fit the selected budget and crew constraints. Increase the budget or crew count.")
            with st.expander("View full maintenance review queue"):
                queue = work.sort_values("Priority score", ascending=False)
                st.dataframe(queue, use_container_width=True, hide_index=True, height=400)
                st.download_button("Download full maintenance queue (CSV)", queue.to_csv(index=False).encode("utf-8"), "jalshield_maintenance_queue.csv", "text/csv")
        else:
            st.warning("The loaded dataset does not contain Priority_Score, so a dispatch manifest cannot be calculated.")
    else:
        st.info("Add predictive_output.csv or fused_tn_pumps.csv to build the maintenance queue.")

elif page == "Live Sensors":
    import json, time
    from datetime import datetime
    st.markdown("### USB hardware connection")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**ESP32 → USB cable → laptop → JalShield dashboard**")
    st.caption("Connect the ESP32 by USB. Close Arduino Serial Monitor before reading here, because only one program can usually use the serial port at a time.")
    try:
        import serial
        from serial.tools import list_ports
        ports = list(list_ports.comports())
        port_options = {f"{item.device} — {item.description}": item.device for item in ports}
        if not port_options:
            st.warning("No serial ports detected. Connect the ESP32 with a USB data cable, install its USB driver if needed, then refresh this page.")
        else:
            chosen_label = st.selectbox("ESP32 serial port", list(port_options.keys()), key="usb_serial_port")
            baud = st.selectbox("Baud rate", [115200, 9600, 57600, 38400], index=0, key="usb_baud")
            if st.button("Read USB sensor values", key="read_usb_sensors", type="primary"):
                latest = None
                errors = []
                try:
                    with serial.Serial(port_options[chosen_label], int(baud), timeout=0.35) as ser:
                        time.sleep(1.8)
                        deadline = time.time() + 4.0
                        while time.time() < deadline:
                            raw = ser.readline().decode("utf-8", errors="ignore").strip()
                            if raw.startswith("{") and raw.endswith("}"):
                                try:
                                    candidate = json.loads(raw)
                                    if isinstance(candidate, dict) and any(k in candidate for k in ("vibration_ms2", "tds_ppm", "temp_c", "flow_lpm")):
                                        latest = candidate
                                except json.JSONDecodeError:
                                    pass
                    if latest is not None:
                        latest["timestamp"] = time.time()
                        latest["source"] = "USB serial"
                        (ROOT / "live_hardware_data.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
                        st.session_state["jalshield_usb_latest"] = latest
                        st.success("USB readings received successfully.")
                    else:
                        st.error("No JSON sensor packet received. Upload the USB firmware supplied with this dashboard and try again.")
                except Exception as exc:
                    st.error(f"Could not read {port_options[chosen_label]}: {exc}")
            live = st.session_state.get("jalshield_usb_latest")
            if live is None:
                live_path = ROOT / "live_hardware_data.json"
                if live_path.exists():
                    try:
                        candidate = json.loads(live_path.read_text(encoding="utf-8"))
                        if candidate.get("source") == "USB serial":
                            live = candidate
                    except Exception:
                        pass
            if live:
                st.success("USB DATA · Latest successful reading")
                a,b,c = st.columns(3)
                a.metric("Acceleration magnitude", f"{live.get('vibration_ms2', '—')} m/s²")
                b.metric("TDS (uncalibrated)", f"{live.get('tds_ppm', '—')} ppm")
                c.metric("Temperature", f"{live.get('temp_c', '—')} °C")
                d,e,f = st.columns(3)
                d.metric("Flow rate", f"{live.get('flow_lpm', '—')} L/min")
                e.metric("PIS screening score", f"{live.get('pis_score', 'Not supplied')}")
                f.metric("Alert status", str(live.get('status', 'SENSOR DATA ONLY')))
                st.caption("Flow is simulated until a physical flow sensor is installed. TDS is an approximate, uncalibrated prototype value. PIS/status are not validated unless calculated by a separate model.")
                with st.expander("Raw USB payload"):
                    st.json(live)
                st.download_button("Download latest readings (JSON)", data=json.dumps(live, indent=2), file_name="jalshield_usb_readings.json", mime="application/json")
    except ImportError:
        st.error("The pyserial package is missing. Run: python -m pip install pyserial")
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("USB integration is manual: press Read USB sensor values whenever you want to capture the latest readings.")

elif page == "Data Explorer":
    st.markdown("### Explore the current dataset")
    if not df.empty:
        a,b,c = st.columns(3)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", f"{len(df.columns):,}")
        c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        query = st.text_input("Find a column", placeholder="e.g. groundwater, district, priority")
        cols = [c for c in df.columns if query.lower() in c.lower()] if query else list(df.columns)
        st.write(f"Showing {len(cols)} columns")
        st.dataframe(df[cols].head(250), use_container_width=True, hide_index=True)
        st.download_button("Download loaded dataset", data=df.to_csv(index=False).encode("utf-8"), file_name="jalshield_current_data.csv", mime="text/csv")
    else:
        st.info("Add predictive_output.csv or fused_tn_pumps.csv to the project folder.")


st.markdown('<div class="footer-note">JalShield · A calmer way to care for water infrastructure · Prototype interface</div>', unsafe_allow_html=True)
