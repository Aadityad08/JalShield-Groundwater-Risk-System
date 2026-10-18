from pathlib import Path
import html
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster

ROOT = Path(__file__).parent
DATA_PATHS = [
    ROOT / "predictive_output.csv",
    ROOT / "fused_tn_pumps.csv",
]
HOTSPOT_PATH = ROOT / "tamilnadu_hotspot_clusters.csv"

# Data loading and geospatial helpers
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

def map_frame(mode, df):
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
    # A fixed Tamil Nadu frame prevents a wide or sparse dataset from opening on the wrong coast.
    fmap.fit_bounds([[7.8, 76.0], [13.7, 80.7]], padding=(12, 12))
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
