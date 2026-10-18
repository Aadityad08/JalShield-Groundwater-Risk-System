#!/usr/bin/env python3
"""
predictive_engine.py - Tamil Nadu Water Infrastructure Intelligence System
==========================================================================

Decision-support PROTOTYPE that reads the fused dataset produced by
data_fusion.py and produces:

  predictive_output.csv               baseline + 2031 scenario scores, preserved metadata
  tamilnadu_water_hotspots.html       interactive point map (markers, thematic layers, heatmap, clusters)
  tamilnadu_district_risk_map.html    district choropleth (only if a genuine boundary file is found)
  tamilnadu_hotspot_clusters.csv      DBSCAN (haversine) clusters of high-priority locations
  validation_report.txt               copy of the console validation report

Install:  pip install pandas numpy scikit-learn xgboost folium geopandas

Usage:
  python predictive_engine.py fused_tn_pumps.csv \
      --sensor-map vibration=sensor_02 flow_rate=sensor_04 --allow-unverified-sensors \
      --boundary districts.kml --outdir out

IMPORTANT LIMITATIONS (repeated in CSV columns, popups, legends and the report)
  1. Coordinates are usually water-body / monitoring-station PROXIES, not verified pumps.
  2. Sensor values, chemistry, population, chlorine and machine-status labels may be SYNTHETIC.
  3. JJM coverage is a DISTRICT-level aggregate; missing values stay missing.
  4. Groundwater readings belong to monitoring stations, not to the mapped coordinate.
  5. 2031 values and monsoon deviations are SCENARIOS, not validated forecasts.
  6. Scores are screening indices - NOT calibrated probabilities.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Dependency check (required vs optional)
# ----------------------------------------------------------------------------
_missing = []
for _mod, _pip in [("pandas", "pandas"), ("numpy", "numpy"), ("sklearn", "scikit-learn"),
                   ("xgboost", "xgboost")]:
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pip)
if _missing:
    sys.exit("ERROR: missing required packages: " + ", ".join(_missing) +
             "\nInstall with:  pip install pandas numpy scikit-learn xgboost folium geopandas")

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CANONICAL_FEATURES = ["vibration", "flow_rate", "motor_temp", "Fluoride_Level",
                      "Nitrate_Level", "TDS", "WQI", "CCI", "lpcd_draw", "residual_chlorine"]
STATUS_NAMES = {0: "NORMAL", 1: "RECOVERING", 2: "BROKEN"}
PRESERVE_FIELDS = ["latitude", "longitude", "district", "groundwater_station",
                   "groundwater_level_m", "groundwater_observation_time",
                   "groundwater_station_distance_km", "jjm_total_households",
                   "jjm_households_with_tap", "jjm_households_without_tap", "jjm_tap_coverage_pct"]
SCENARIO_NOTE = "SCENARIO/SIMULATION - not a validated forecast or measurement"


@dataclass
class Config:
    """All tunable, documented parameters in one place."""
    # model
    n_estimators: int = 200          # boosting rounds (small: data are small / synthetic)
    max_depth: int = 3               # shallow trees limit over-fitting on ~1000 rows
    learning_rate: float = 0.1
    test_size: float = 0.25
    min_class_for_split: int = 4     # need >= this many rows per class for stratified split
    # chemistry screening thresholds (project-defined; fluoride/nitrate/TDS follow common
    # drinking-water guideline values: BIS 1.5 mg/L F, 45 mg/L NO3, 2000 mg/L TDS upper permissible)
    fluoride_threshold: float = 1.5
    nitrate_threshold: float = 45.0
    tds_threshold: float = 2000.0
    wqi_concern_below: float = 50.0  # WQI <= this => full concern contribution
    # contamination screening weights (sum to 1)
    w_fluoride: float = 0.35
    w_nitrate: float = 0.25
    w_tds: float = 0.20
    w_wqi: float = 0.20
    # constraint thresholds (project-defined, NOT regulatory)
    chlorine_low: float = 0.2
    chlorine_high: float = 0.5
    lpcd_low: float = 55.0
    # priority score
    chlorine_penalty: float = 0.15   # +15 % score if chlorine anomaly
    low_supply_penalty: float = 0.15 # +15 % score if low supply
    high_priority_quantile: float = 0.80   # top 20 % => High
    medium_priority_quantile: float = 0.50 # next 30 % => Medium
    # growth scenario
    fluoride_growth: float = 1.05
    tds_growth: float = 1.08
    scenario_years: int = 5          # 2026 -> 2031
    base_year: int = 2026
    rul_cap_months: float = 120.0
    # climate scenario
    monsoon_min: float = -40.0
    monsoon_max: float = 20.0
    drought_limit_pct: float = 40.0  # -40 % deviation => drought severity 1
    nitrate_norm_max: float = 90.0   # 2 x threshold => agri-pollution score 1
    cvi_w_drought: float = 0.40
    cvi_w_nitrate: float = 0.30
    cvi_w_groundwater: float = 0.30
    climate_max_adjustment: float = 0.25  # score multiplied by up to 1.25
    gw_max_station_km: float = 15.0  # station must be within this distance of the mapped point
    gw_plausible_abs_m: float = 100.0  # |level| above this treated as implausible
    gw_sentinel_below: float = -900.0  # e.g. -999.999 no-data sentinel
    gw_max_obs_span_days: int = 366  # observations must lie within this window of the newest one
    # geography
    tn_bbox: tuple = (7.9, 13.7, 76.1, 80.5)  # lat_min, lat_max, lon_min, lon_max
    # clustering
    cluster_radius_km: float = 25.0
    cluster_min_samples: int = 5
    cluster_rule: str = "High"       # "High" category, or "quantile:0.9"
    seed: int = 42
    heat_note: str = ("Heatmap = density of mapped REFERENCE locations weighted by priority "
                      "(min-max normalised); it is not a density of verified pumps.")


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
DISTRICT_ALIASES = {
    "the nilgiris": "nilgiris", "nilgiri": "nilgiris", "n|lgiris": "nilgiris",
    "kanyakumari": "kanniyakumari", "villupuram": "viluppuram", "villuppuram": "viluppuram",
    "thoothukkudi": "thoothukudi", "tuticorin": "thoothukudi",
    "thiruvallur": "tiruvallur", "thirunelveli": "tirunelveli",
    "thiruchirappalli": "tiruchirappalli", "tiruchirapalli": "tiruchirappalli",
    "trichy": "tiruchirappalli", "tiruchirappalli": "tiruchirappalli",
    "thiruvarur": "tiruvarur", "kanchipuram": "kancheepuram",
    "sivagangai": "sivaganga", "tirupathur": "tirupathur", "tiruppattur": "tirupathur",
    "kallakkurichi": "kallakurichi", "ranippettai": "ranipet", "teni": "theni",
    "thiruvannamalai": "tiruvannamalai", "madras": "chennai", "tirunelveli": "tirunelveli",
    "pudukottai": "pudukkottai", "mayiladuthurai": "mayiladuthurai",
}


def norm_district(x) -> str:
    """Normalised join key for district names (original names are kept elsewhere)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = re.sub(r"\s+", " ", str(x).strip()).casefold()
    return DISTRICT_ALIASES.get(s, s)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def fmt(v, nd=2, unit=""):
    """Format a value for display; missing -> 'unavailable' (never zero)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return "unavailable"
    if isinstance(v, (int, np.integer)):
        return f"{v}{unit}"
    try:
        return f"{float(v):.{nd}f}{unit}"
    except (TypeError, ValueError):
        return str(v)


class Report:
    """Collects console output so it can also be written to validation_report.txt."""
    def __init__(self):
        self.lines: list[str] = []
        self.warnings: list[str] = []

    def p(self, text=""):
        print(text)
        self.lines.append(str(text))

    def warn(self, text):
        self.warnings.append(text)

    def head(self, title):
        self.p("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ----------------------------------------------------------------------------
# Objective 1 - load / validate / prepare
# ----------------------------------------------------------------------------
def parse_sensor_map(items):
    """Parse ['vibration=sensor_02', 'flow_rate=sensor_04'] into a dict."""
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"ERROR: --sensor-map entries must look like canonical=column, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_data(path: Path, rep: Report) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: input file not found: {path}\n"
                 "Run data_fusion.py first or pass the path as the first argument.")
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ERROR: cannot read {path}: {e}")
    if df.empty:
        sys.exit("ERROR: input CSV has no rows.")
    rep.p(f"Loaded {path} -> {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def resolve_sensor_mapping(df, user_map, allow_unverified, rep) -> dict:
    """
    Map canonical features vibration / flow_rate / motor_temp to source columns.

    A sensor column is NEVER assumed to measure a physical quantity. Rules:
      * motor_temp  -> 'sensor_10_motor_temp' (name states the meaning) unless overridden.
      * vibration   -> must be named via --sensor-map (no column in the fused file is named
                       as vibration).
      * flow_rate   -> 'sensor_04' only if the user confirms it via --sensor-map.
    Mappings supplied by the user that the file itself cannot verify are only accepted when
    --allow-unverified-sensors is given; they are then recorded as UNVERIFIED everywhere.
    """
    mapping, verified = {}, {}
    if "sensor_10_motor_temp" in df.columns:
        mapping["motor_temp"], verified["motor_temp"] = "sensor_10_motor_temp", True
    for key, col in user_map.items():
        if key not in ("vibration", "flow_rate", "motor_temp"):
            sys.exit(f"ERROR: --sensor-map key '{key}' not one of vibration, flow_rate, motor_temp")
        if col not in df.columns:
            sys.exit(f"ERROR: --sensor-map {key}={col}: column '{col}' not in input.")
        mapping[key] = col
        name_ok = (key == "vibration" and "vibrat" in col.lower()) or \
                  (key == "flow_rate" and "flow" in col.lower()) or \
                  (key == "motor_temp" and "temp" in col.lower())
        verified[key] = name_ok
    # auto-detect vibration columns named as such
    if "vibration" not in mapping:
        cands = [c for c in df.columns if "vibrat" in c.lower()]
        if len(cands) == 1:
            mapping["vibration"], verified["vibration"] = cands[0], True
    if "flow_rate" not in mapping:
        cands = [c for c in df.columns if "flow" in c.lower()]
        if len(cands) == 1:
            mapping["flow_rate"], verified["flow_rate"] = cands[0], True

    missing = [k for k in ("vibration", "flow_rate", "motor_temp") if k not in mapping]
    if missing:
        sys.exit(
            "ERROR: cannot verify sensor meaning for: " + ", ".join(missing) + ".\n"
            "  The fused file has generic columns (sensor_00 ... sensor_51) and no documented\n"
            "  physical meaning. Supply the mapping you have verified, e.g.\n"
            "      --sensor-map vibration=<column> flow_rate=sensor_04\n"
            "  (use only columns whose physical quantity you have confirmed from the sensor\n"
            "  documentation).")
    unverified = [k for k, v in verified.items() if not v]
    if unverified and not allow_unverified:
        sys.exit(
            "ERROR: sensor mapping for " + ", ".join(unverified) + " is not verifiable from the\n"
            "  input file (column names do not state the quantity).\n"
            "  If you have confirmed the mapping externally re-run with --allow-unverified-sensors;\n"
            "  otherwise supply a verified column. Refusing to guess.")
    if unverified:
        rep.warn("UNVERIFIED sensor mapping used (user-asserted, --allow-unverified-sensors): " +
                 ", ".join(f"{k}<-{mapping[k]}" for k in unverified))
    return {"map": mapping, "unverified": unverified}


def prepare_frame(df, sensors, cfg, rep) -> pd.DataFrame:
    """Create canonical numeric columns and data-quality flags. Source columns untouched."""
    out = df.copy()
    out["district_key"] = out["district"].map(norm_district)
    for canon, col in sensors["map"].items():
        out[canon] = pd.to_numeric(out[col], errors="coerce")
        out[f"{canon}_source_column"] = col
    for c in ["Fluoride_Level", "Nitrate_Level", "TDS", "WQI", "CCI", "lpcd_draw",
              "residual_chlorine", "population_served", "latitude", "longitude",
              "groundwater_level_m", "groundwater_station_distance_km"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # Non-finite -> missing (keeps rows; never fabricates values)
    for c in CANONICAL_FEATURES:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    # -------- data-quality provenance flags --------
    src = lambda c: out[c].astype(str).str.lower() if c in out else pd.Series("", index=out.index)
    out["sensor_values_synthetic"] = src("sensor_data_source").str.contains("synthetic")
    # chemistry is treated as synthetic unless the source column says otherwise
    out["chemistry_values_synthetic"] = ~src("water_chemistry_source").str.contains("lab|cgwb observation|measured") \
        if "water_chemistry_source" in out else True
    # data_fusion.py draws population / chlorine from an RNG, so treat them as synthetic
    # whenever the sensor/chemistry provenance is synthetic or unknown.
    out["population_chlorine_synthetic"] = out["sensor_values_synthetic"] | out["chemistry_values_synthetic"]
    out["machine_status_synthetic"] = out["sensor_values_synthetic"]
    loc = src("location_source")
    out["geospatial_data_quality"] = np.select(
        [loc.str.contains("verified pump|verified asset|surveyed pump"),
         loc.str.contains("water-body|waterbody"),
         loc.str.contains("monitoring station")],
        ["verified_asset", "geographic_proxy_waterbody", "geographic_proxy_gw_station"],
        default="unknown_provenance")
    # -------- coordinates --------
    lat_ok = out["latitude"].between(-90, 90) & out["longitude"].between(-180, 180)
    out["coords_valid"] = lat_ok & out["latitude"].notna() & out["longitude"].notna()
    la0, la1, lo0, lo1 = cfg.tn_bbox
    out["coords_in_tn_bbox"] = out["coords_valid"] & out["latitude"].between(la0, la1) & \
        out["longitude"].between(lo0, lo1)
    # -------- groundwater: sentinel / plausibility / recency / distance --------
    gw = out["groundwater_level_m"]
    out["gw_sentinel_value"] = gw <= cfg.gw_sentinel_below
    out["gw_implausible_value"] = (gw.abs() > cfg.gw_plausible_abs_m) & ~out["gw_sentinel_value"]
    obs = pd.to_datetime(out.get("groundwater_observation_time"), errors="coerce")
    newest = obs.max()
    out["gw_obs_recent"] = obs.notna() & ((newest - obs).dt.days <= cfg.gw_max_obs_span_days)
    out["gw_station_near"] = out["groundwater_station_distance_km"] <= cfg.gw_max_station_km
    out["gw_usable"] = gw.notna() & ~out["gw_sentinel_value"] & ~out["gw_implausible_value"] & \
        out["gw_obs_recent"] & out["gw_station_near"]
    return out


def validate_inputs(df, rep):
    needed = ["latitude", "longitude", "district", "status_group", "Fluoride_Level",
              "Nitrate_Level", "TDS", "WQI", "CCI", "lpcd_draw", "residual_chlorine",
              "population_served"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        sys.exit("ERROR: input file lacks required columns: " + ", ".join(miss))
    meta_missing = [c for c in PRESERVE_FIELDS if c not in df.columns]
    if meta_missing:
        rep.warn("Optional metadata columns absent from input: " + ", ".join(meta_missing))


# ----------------------------------------------------------------------------
# Objective 2 - imbalance-aware XGBoost
# ----------------------------------------------------------------------------
def class_weights(y: np.ndarray) -> dict:
    counts = np.bincount(y, minlength=3)
    n, k = len(y), int((counts > 0).sum())
    return {c: (n / (k * counts[c]) if counts[c] > 0 else 0.0) for c in range(len(counts))}


def make_model(cfg) -> XGBClassifier:
    return XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                         n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
                         learning_rate=cfg.learning_rate, random_state=cfg.seed,
                         n_jobs=1, tree_method="hist", verbosity=0)


def train_and_evaluate(df, cfg, rep):
    """Evaluate on a stratified hold-out, then refit on all valid labelled rows."""
    labelled = df["status_group"].isin([0, 1, 2])
    if (~labelled).any():
        rep.warn(f"{int((~labelled).sum())} rows with missing/invalid status_group excluded from TRAINING "
                 "(retained in output with predictions).")
    tr = df[labelled].copy()
    y = tr["status_group"].astype(int).to_numpy()
    counts = np.bincount(y, minlength=3)
    rep.p("Class distribution (NORMAL/RECOVERING/BROKEN): " + str(counts.tolist()))
    if (counts == 0).any():
        sys.exit("ERROR: a status class has zero examples (counts=" + str(counts.tolist()) +
                 "); a 3-class model cannot be trained honestly.")
    cw = class_weights(y)
    rep.p("Inverse-frequency class weights: " + str({STATUS_NAMES[k]: round(float(v), 3) for k, v in cw.items()}))
    X = tr[CANONICAL_FEATURES]
    metrics = {"class_counts": counts.tolist(), "class_weights": cw}
    notes = []
    if counts.min() >= cfg.min_class_for_split and len(tr) >= 50:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, stratify=y,
                                              random_state=cfg.seed)
        imp = SimpleImputer(strategy="median").fit(Xtr)  # fitted on TRAINING data only
        sw = np.array([cw[c] for c in ytr])
        m = make_model(cfg).fit(imp.transform(Xtr), ytr, sample_weight=sw)
        pred = m.predict(imp.transform(Xte))
        labels = [0, 1, 2]
        metrics.update(
            macro_f1=f1_score(yte, pred, average="macro", labels=labels, zero_division=0),
            balanced_accuracy=balanced_accuracy_score(yte, pred),
            report=classification_report(yte, pred, labels=labels,
                                         target_names=list(STATUS_NAMES.values()), zero_division=0),
            confusion=confusion_matrix(yte, pred, labels=labels), n_test=len(yte),
            test_counts=np.bincount(yte, minlength=3).tolist())
        if min(metrics["test_counts"]) < 10:
            notes.append(f"Hold-out contains only {metrics['test_counts']} examples per class; "
                         "per-class metrics are statistically unstable.")
    else:
        notes.append("Too few examples per class for a stratified hold-out; NO performance metrics "
                     "reported (none fabricated).")
        metrics["macro_f1"] = metrics["balanced_accuracy"] = np.nan
    # --- final model on all valid labelled rows ---
    imp = SimpleImputer(strategy="median").fit(X)
    sw = np.array([cw[c] for c in y])
    final = make_model(cfg).fit(imp.transform(X), y, sample_weight=sw)
    experimental = bool(df["machine_status_synthetic"].any())
    if experimental:
        notes.append("EXPERIMENTAL MODEL: machine-status labels and sensor values are SYNTHETIC. "
                     "Outputs are not validated failure probabilities.")
    if not np.isnan(metrics["macro_f1"]) and experimental:
        notes.append("Evaluation is NOT statistically meaningful: synthetic labels have no verified "
                     "relationship to the features; any score reflects the generator, not real pumps.")
    metrics["notes"] = notes
    metrics["experimental"] = experimental
    return final, imp, metrics


def print_metrics(m, rep):
    rep.head("MODEL EVALUATION (stratified hold-out; EXPERIMENTAL)")
    if "report" in m:
        rep.p(f"Hold-out rows: {m['n_test']}  class counts: {m['test_counts']}")
        rep.p(f"Macro-F1: {m['macro_f1']:.3f}   Balanced accuracy: {m['balanced_accuracy']:.3f}")
        rep.p("Classification report:\n" + m["report"])
        rep.p("Confusion matrix (rows=true NORMAL/RECOVERING/BROKEN, cols=predicted):\n" + str(m["confusion"]))
    for n in m["notes"]:
        rep.p("NOTE: " + n)


def predict_probs(model, imputer, frame):
    """Model probabilities using the SAME fitted imputer and feature order."""
    Xi = imputer.transform(frame[CANONICAL_FEATURES])
    if not np.isfinite(Xi).all():
        raise ValueError("Non-finite values after imputation.")
    return model.predict_proba(Xi)


# ----------------------------------------------------------------------------
# Objective 3/4/5 - screening scores, constraints, priority
# ----------------------------------------------------------------------------
def contamination_score(f, n, t, w, cfg):
    """
    Water_Contamination_Risk (0-1 screening index, NOT a probability):
      component_i = clip(value / threshold, 0, 1)   for fluoride, nitrate, TDS  (1 at threshold)
      wqi_comp    = clip((100 - WQI) / (100 - cfg.wqi_concern_below), 0, 1)
      score       = weighted mean of available components with fixed weights; NaN if ANY of the
                    four inputs is missing (unavailable != safe).
    """
    comp = pd.DataFrame({
        "f": np.clip(f / cfg.fluoride_threshold, 0, 1),
        "n": np.clip(n / cfg.nitrate_threshold, 0, 1),
        "t": np.clip(t / cfg.tds_threshold, 0, 1),
        "w": np.clip((100 - w) / (100 - cfg.wqi_concern_below), 0, 1)})
    score = (cfg.w_fluoride * comp["f"] + cfg.w_nitrate * comp["n"] +
             cfg.w_tds * comp["t"] + cfg.w_wqi * comp["w"])
    return score.where(comp.notna().all(axis=1))


def constraint_flags(df, cfg):
    """Chlorine / low-supply flags. Missing readings stay unknown (flag False + missing flag True)."""
    rc, lp = df["residual_chlorine"], df["lpcd_draw"]
    df["chlorine_data_missing"] = rc.isna()
    df["lpcd_data_missing"] = lp.isna()
    df["chlorine_anomaly"] = (rc < cfg.chlorine_low) | (rc > cfg.chlorine_high)
    df["low_supply"] = lp < cfg.lpcd_low
    df["chlorine_status"] = np.select([rc.isna(), df["chlorine_anomaly"]], ["unknown", "anomaly"], "in_range")
    df["supply_status"] = np.select([lp.isna(), df["low_supply"]], ["unknown", "low"], "adequate")
    return df


def priority_base(mfr, wcr, cci, pop):
    """Priority_Score_Base = (MFR*0.55 + WCR*0.25) * (CCI*0.20) * log1p(pop). NaN if any input NaN."""
    return (mfr * 0.55 + wcr * 0.25) * (cci * 0.20) * np.log1p(pop)


def apply_penalties(base, chl_anom, low_sup, cfg):
    mult = 1 + cfg.chlorine_penalty * chl_anom.astype(float) + cfg.low_supply_penalty * low_sup.astype(float)
    return base * mult, mult


def categorise(scores: pd.Series, hi_cut: float, med_cut: float) -> pd.Series:
    """Intervention category from fixed cut-points (shared by CSV, maps and validation)."""
    cat = pd.Series("Unrated (insufficient data)", index=scores.index, dtype="object")
    ok = scores.notna()
    cat[ok & (scores < med_cut)] = "Low"
    cat[ok & (scores >= med_cut) & (scores < hi_cut)] = "Medium"
    cat[ok & (scores >= hi_cut)] = "High"
    return cat


# ----------------------------------------------------------------------------
# Objective 6 - chemical-threshold RUL scenario
# ----------------------------------------------------------------------------
def months_to_breach(baseline: pd.Series, growth: float, threshold: float, cap: float) -> pd.Series:
    """
    Fractional years n solving baseline * growth**n = threshold  ->  n = ln(T/b)/ln(g);
    converted to months. 0 if already at/above threshold. NaN if baseline missing/non-positive
    while below threshold cannot be computed. Capped at `cap` months.
    """
    b = baseline.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.log(threshold / b) / np.log(growth)
    months = n * 12
    months = months.where(b < threshold, 0.0)           # already breached -> 0
    months = months.where(b.notna())                     # missing stays missing
    months = months.where(~(b <= 0) | b.isna(), cap)     # non-positive baseline never breaches
    return months.clip(0, cap)


def rul_block(df, cfg):
    """Annual projections to 2031 + threshold times. Fully scenario-based."""
    for n in range(1, cfg.scenario_years + 1):
        yr = cfg.base_year + n
        df[f"Fluoride_{yr}"] = df["Fluoride_Level"] * cfg.fluoride_growth ** n
        df[f"TDS_{yr}"] = df["TDS"] * cfg.tds_growth ** n
        if n > 1:
            df[f"Fluoride_change_{yr}"] = df[f"Fluoride_{yr}"] - df[f"Fluoride_{yr - 1}"]
            df[f"TDS_change_{yr}"] = df[f"TDS_{yr}"] - df[f"TDS_{yr - 1}"]
        else:
            df[f"Fluoride_change_{yr}"] = df[f"Fluoride_{yr}"] - df["Fluoride_Level"]
            df[f"TDS_change_{yr}"] = df[f"TDS_{yr}"] - df["TDS"]
    last = cfg.base_year + cfg.scenario_years
    # first projected calendar year each threshold is reached (NaN if not within horizon)
    for name, thr in (("Fluoride", cfg.fluoride_threshold), ("TDS", cfg.tds_threshold)):
        yrs = pd.DataFrame({y: df[f"{name}_{y}"] >= thr for y in range(cfg.base_year + 1, last + 1)})
        first = yrs.idxmax(axis=1).where(yrs.any(axis=1))
        first = first.where(df[name if name == "TDS" else "Fluoride_Level"].notna())
        base_hit = df["Fluoride_Level" if name == "Fluoride" else "TDS"] >= thr
        df[f"{name}_first_breach_year"] = first.where(~base_hit, cfg.base_year)
    df["Fluoride_RUL_Months"] = months_to_breach(df["Fluoride_Level"], cfg.fluoride_growth,
                                                 cfg.fluoride_threshold, cfg.rul_cap_months)
    df["TDS_RUL_Months"] = months_to_breach(df["TDS"], cfg.tds_growth, cfg.tds_threshold, cfg.rul_cap_months)
    both = df[["Fluoride_RUL_Months", "TDS_RUL_Months"]]
    df["Aquifer_RUL_Months"] = both.min(axis=1, skipna=False).clip(0, cfg.rul_cap_months)
    lim = np.where(both["Fluoride_RUL_Months"] < both["TDS_RUL_Months"], "Fluoride",
                   np.where(both["TDS_RUL_Months"] < both["Fluoride_RUL_Months"], "TDS", "Fluoride=TDS"))
    lim = pd.Series(lim, index=df.index, dtype="object")
    lim[(both >= cfg.rul_cap_months).all(axis=1)] = "None within 10 years (capped)"
    lim[both.isna().any(axis=1)] = "unavailable"
    df["Limiting_Chemical"] = lim
    df["RUL_label"] = "SCENARIO chemical-threshold indicator (5%/yr F, 8%/yr TDS); not aquifer lifetime"
    return df


# ----------------------------------------------------------------------------
# Objective 7 - climate scenario
# ----------------------------------------------------------------------------
def groundwater_stress(df, cfg):
    """
    Relative groundwater stress (0-1) among usable readings.
    Assumption (UNVERIFIED, source gives no datum): the telemetry reports metres with NEGATIVE values
    = depth below ground, so deeper (more negative) = higher stress. Positive / sentinel /
    implausible / stale / far-station readings are excluded. Stress = percentile rank of depth among
    usable rows, i.e. a RELATIVE ranking, not an absolute depletion measure.
    """
    usable = df["gw_usable"] & (df["groundwater_level_m"] < 0)
    depth = (-df["groundwater_level_m"]).where(usable)
    return depth.rank(pct=True)


def climate_block(df, cfg, rng_seed):
    rng = np.random.default_rng(rng_seed)
    df["monsoon_deviation_pct"] = rng.uniform(cfg.monsoon_min, cfg.monsoon_max, len(df)).round(2)
    df["monsoon_deviation_source"] = "SYNTHETIC scenario (seeded uniform); not observed rainfall"
    drought = np.clip(-df["monsoon_deviation_pct"] / cfg.drought_limit_pct, 0, 1)
    agri = np.clip(df["Nitrate_Level"] / cfg.nitrate_norm_max, 0, 1)
    gws = groundwater_stress(df, cfg)
    df["cvi_drought_component"], df["cvi_nitrate_component"], df["cvi_groundwater_component"] = drought, agri, gws
    w = pd.DataFrame({"d": cfg.cvi_w_drought * drought.notna(), "n": cfg.cvi_w_nitrate * agri.notna(),
                      "g": cfg.cvi_w_groundwater * gws.notna()})
    num = (cfg.cvi_w_drought * drought.fillna(0) + cfg.cvi_w_nitrate * agri.fillna(0) +
           cfg.cvi_w_groundwater * gws.fillna(0))
    # weights renormalised over AVAILABLE components; recorded in cvi_components_used
    df["Climate_Vulnerability_Index"] = (100 * num / w.sum(axis=1)).where(w.sum(axis=1) > 0).clip(0, 100)
    df["cvi_components_used"] = np.select(
        [gws.isna() & agri.isna(), gws.isna()], ["drought only", "drought+nitrate (no usable groundwater)"],
        "drought+nitrate+groundwater")
    df["Climate_Adjustment_Factor"] = 1 + cfg.climate_max_adjustment * (df["Climate_Vulnerability_Index"] / 100)
    return df


# ----------------------------------------------------------------------------
# Objective 8 - 2031 scenario
# ----------------------------------------------------------------------------
@dataclass
class WQIModel:
    """Baseline WQI formula from data_fusion.py with FROZEN min/max (no refit on future data)."""
    mins: np.ndarray
    maxs: np.ndarray

    def __call__(self, f, n, t):
        chem = np.column_stack([f, n, t]).astype(float)
        scaled = (chem - self.mins) / (self.maxs - self.mins)
        pollution = .40 * scaled[:, 0] + .35 * scaled[:, 1] + .25 * scaled[:, 2]
        wqi = 100 * (1 - pollution) - 20 * (chem[:, 0] > 1.5) - 20 * (chem[:, 1] > 45)
        wqi = np.clip(wqi, 0, 100)
        wqi[~np.isfinite(chem).all(axis=1)] = np.nan
        return wqi


def future_block(df, cfg, model, imputer, wqi_model, hi_cut, med_cut, rep):
    yr = cfg.base_year + cfg.scenario_years  # 2031
    fut = df.copy()
    fut["Fluoride_Level"] = df[f"Fluoride_{yr}"]
    fut["TDS"] = df[f"TDS_{yr}"]
    fut["WQI"] = wqi_model(fut["Fluoride_Level"], fut["Nitrate_Level"], fut["TDS"])
    fut["CCI"] = (fut["TDS"] / 1000) * (df["motor_temp"] / 40)    # same formula as data_fusion.py
    # vibration, flow_rate, motor_temp, nitrate, lpcd, chlorine: ASSUMED UNCHANGED
    P = predict_probs(model, imputer, fut)
    out = pd.DataFrame(index=df.index)
    out["Fluoride_2031"], out["TDS_2031"] = fut["Fluoride_Level"], fut["TDS"]
    out["WQI_2031"], out["CCI_2031"] = fut["WQI"], fut["CCI"]
    out["Mechanical_Failure_Risk_2031"] = P[:, 2]
    out["Recovery_or_Maintenance_Risk_2031"] = P[:, 1]
    out["Water_Contamination_Risk_2031"] = contamination_score(
        fut["Fluoride_Level"], fut["Nitrate_Level"], fut["TDS"], fut["WQI"], cfg)
    base31 = priority_base(out["Mechanical_Failure_Risk_2031"], out["Water_Contamination_Risk_2031"],
                           out["CCI_2031"], df["population_served"])
    pen31, _ = apply_penalties(base31, df["chlorine_anomaly"], df["low_supply"], cfg)
    out["Priority_Score_Base_2031"] = base31
    out["Priority_Score_2031"] = pen31 * df["Climate_Adjustment_Factor"]   # same climate factor held fixed
    out["Intervention_Category_2031_fixed_cuts"] = categorise(out["Priority_Score_2031"], hi_cut, med_cut)
    out["future_assumptions"] = ("SCENARIO 2031: F +5%/yr, TDS +8%/yr; WQI/CCI recomputed with frozen baseline "
                                 "scaling; vibration, flow, motor temp, nitrate, LPCD, chlorine, population, "
                                 "monsoon deviation ASSUMED UNCHANGED; not a validated forecast")
    return out


# ----------------------------------------------------------------------------
# Objective 13 - DBSCAN clustering
# ----------------------------------------------------------------------------
def select_cluster_points(df, cfg):
    if cfg.cluster_rule.startswith("quantile:"):
        q = float(cfg.cluster_rule.split(":")[1])
        thr = df["Priority_Score"].quantile(q)
        return df["coords_valid"] & df["Priority_Score"].ge(thr)
    return df["coords_valid"] & (df["Intervention_Category"] == "High")


def run_clustering(df, cfg, rep):
    df["Hotspot_Cluster_ID"] = np.nan
    df["Hotspot_Cluster_Label"] = "not eligible (not high priority / invalid coords)"
    sel = select_cluster_points(df, cfg)
    n = int(sel.sum())
    rep.p(f"Clustering: {n} eligible high-priority points; radius={cfg.cluster_radius_km} km, "
          f"min_samples={cfg.cluster_min_samples}")
    if n < max(cfg.cluster_min_samples, 2):
        rep.p("Clustering SKIPPED: too few eligible points (no clusters fabricated).")
        rep.warn("Hotspot clustering skipped (too few high-priority points).")
        return df, None
    coords = np.radians(df.loc[sel, ["latitude", "longitude"]].to_numpy())
    eps = cfg.cluster_radius_km / 6371.0088   # km -> radians on unit sphere
    lab = DBSCAN(eps=eps, min_samples=cfg.cluster_min_samples, metric="haversine",
                 algorithm="ball_tree").fit_predict(coords)
    df.loc[sel, "Hotspot_Cluster_ID"] = lab
    df.loc[sel, "Hotspot_Cluster_Label"] = np.where(lab == -1, "noise (unclustered)", "clustered")
    sizes = pd.Series(lab[lab >= 0]).value_counts().sort_index()
    rep.p(f"Clusters found: {len(sizes)}; unclustered (noise): {int((lab == -1).sum())}")
    if len(sizes):
        rep.p("Cluster sizes: " + ", ".join(f"#{k}:{v}" for k, v in sizes.items()))
    rep.p("Note: a spatial cluster is a concentration warranting investigation; it does NOT show a "
          "common cause or statistical significance.")
    return df, sizes


# ----------------------------------------------------------------------------
# Boundary loading (genuine data only)
# ----------------------------------------------------------------------------
def find_boundary(explicit, search_dirs):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            print(f"WARNING: boundary file {p} not found.")
            return None
        return p
    for d in search_dirs:
        for pat in ("*.geojson", "*.shp", "*.kml"):
            for p in sorted(Path(d).glob(pat)):
                if _looks_like_tn_districts(p):
                    return p
    return None


def _looks_like_tn_districts(p: Path) -> bool:
    try:
        head = p.read_bytes()[:4000].lower()
    except OSError:
        return False
    return p.suffix.lower() in (".shp",) or b"tamil nadu" in head or b"district" in head


def load_boundary(path: Path, rep):
    """Read district polygons with GeoPandas; repair invalid geometry; drop Z. Returns (gdf, name_col)."""
    if not HAS_GPD:
        rep.warn("geopandas not installed - choropleth and polygon validation unavailable.")
        return None, None
    try:
        g = gpd.read_file(path)
    except Exception as e:  # noqa: BLE001
        rep.warn(f"Boundary file {path} could not be read by GeoPandas/GDAL ({e}).")
        return None, None
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    name_col = next((c for c in g.columns if c.lower() in ("district", "dtname", "dist_name", "name_2",
                                                           "district_name", "name")), None)
    if name_col is None or g.empty:
        rep.warn("Boundary file has no recognisable district-name column.")
        return None, None
    from shapely import force_2d, make_valid
    g["geometry"] = g.geometry.apply(lambda geom: force_2d(make_valid(geom)) if geom is not None else geom)
    g["district_key"] = g[name_col].map(norm_district)
    keep = ["district_key", name_col, "geometry"]
    g = g[keep].dissolve(by="district_key", as_index=False, aggfunc="first")
    rep.p(f"Boundary: {path.name} -> {len(g)} district polygons (geometry repaired where invalid; "
          "boundary file provenance: user-supplied)")
    return g, name_col


def boundary_point_checks(df, gdf, rep):
    """Flag points inside the state outline and inside their labelled district polygon."""
    df["coords_in_tn_boundary"] = np.nan
    df["coords_in_labelled_district_polygon"] = np.nan
    if gdf is None:
        return df
    pts = gpd.GeoDataFrame(df.loc[df["coords_valid"], ["district_key"]],
                           geometry=gpd.points_from_xy(df.loc[df["coords_valid"], "longitude"],
                                                       df.loc[df["coords_valid"], "latitude"]), crs=4326)
    state = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
    inside = pts.geometry.within(state)
    df.loc[inside.index, "coords_in_tn_boundary"] = inside.astype(float)
    poly = gdf.set_index("district_key").geometry
    ok = []
    for idx, row in pts.iterrows():
        g = poly.get(row["district_key"])
        ok.append(np.nan if g is None else float(g.contains(row.geometry) or g.touches(row.geometry)))
    df.loc[pts.index, "coords_in_labelled_district_polygon"] = ok
    out_state = int((inside == False).sum())  # noqa: E712
    mism = int(np.nansum(1 - np.array(ok, dtype=float))) if len(ok) else 0
    rep.p(f"Geographic validation: {out_state} points outside the supplied state boundary; "
          f"{mism} points fall outside their labelled district polygon (labels/coordinates NOT altered).")
    if out_state:
        rep.warn(f"{out_state} points lie outside the boundary polygons (flagged, not moved).")
    if mism:
        rep.warn(f"{mism} points are located in a different district polygon than their 'district' label "
                 "(likely because labels in the fused file were assigned independently of coordinates).")
    return df


# ----------------------------------------------------------------------------
# Objective 11 - point map
# ----------------------------------------------------------------------------
COLORS = {"Low": "#2ca02c", "Medium": "#f2c200", "High": "#d62728",
          "Unrated (insufficient data)": "#888888"}


def ramp(v, lo, hi, colors=("#2ca02c", "#f2c200", "#d62728")):
    """Map value in [lo,hi] to a 3-colour ramp; NaN -> grey."""
    if pd.isna(v):
        return "#888888"
    t = float(np.clip((v - lo) / (hi - lo + 1e-12), 0, 1))
    return colors[0] if t < 1 / 3 else colors[1] if t < 2 / 3 else colors[2]


def popup_html(r, cfg):
    warn = []
    if r["geospatial_data_quality"] != "verified_asset":
        warn.append("Coordinate is a geographic PROXY (" + r["geospatial_data_quality"] + "), not a verified pump.")
    if r["sensor_values_synthetic"]:
        warn.append("Sensor values/machine-status labels SYNTHETIC.")
    if r["chemistry_values_synthetic"]:
        warn.append("Water chemistry SYNTHETIC.")
    if r["population_chlorine_synthetic"]:
        warn.append("Population/chlorine SYNTHETIC.")
    if not r["gw_usable"]:
        warn.append("Groundwater reading not usable (missing/sentinel/implausible/stale/far station).")
    if pd.isna(r["jjm_tap_coverage_pct"]):
        warn.append("District JJM coverage not reported.")
    warn.append("Priority/risk values are screening scores, not probabilities; monsoon & 2031 values are scenarios.")
    gw_txt = fmt(r["groundwater_level_m"], 2, " m") if not r["gw_sentinel_value"] else "invalid sentinel value (excluded)"
    rows = [
        ("District", f"{r['district']}"), ("Coordinates", f"{r['latitude']:.5f}, {r['longitude']:.5f}"),
        ("Priority score / category", f"{fmt(r['Priority_Score'], 3)} / {r['Intervention_Category']}"),
        ("Mechanical failure risk (experimental)", fmt(r["Mechanical_Failure_Risk"], 3)),
        ("Water-contamination screening", fmt(r["Water_Contamination_Risk"], 3)),
        ("Fluoride / Nitrate (mg/L)", f"{fmt(r['Fluoride_Level'])} / {fmt(r['Nitrate_Level'])}"),
        ("TDS (mg/L) / WQI", f"{fmt(r['TDS'], 1)} / {fmt(r['WQI'], 1)}"), ("CCI", fmt(r["CCI"], 3)),
        ("Groundwater level (station, unverified datum)", f"{gw_txt} @ {r['groundwater_station']}"),
        ("Station distance", fmt(r["groundwater_station_distance_km"], 1, " km")),
        ("JJM tap coverage (DISTRICT aggregate)", fmt(r["jjm_tap_coverage_pct"], 1, " %")),
        ("Population served (synthetic)", fmt(r["population_served"], 0)),
        ("Aquifer RUL (scenario)", fmt(r["Aquifer_RUL_Months"], 1, " months")),
        ("Climate Vulnerability Index (sim.)", fmt(r["Climate_Vulnerability_Index"], 1)),
        ("Hotspot cluster", "none/ineligible" if pd.isna(r["Hotspot_Cluster_ID"]) else
         ("noise" if r["Hotspot_Cluster_ID"] < 0 else f"#{int(r['Hotspot_Cluster_ID'])}")),
    ]
    tbl = "".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in rows)
    w = "".join(f"<li>{x}</li>" for x in warn)
    return (f"<div style='font-size:12px;width:340px'><table>{tbl}</table>"
            f"<div style='color:#a33;margin-top:4px'><b>Data-quality warnings</b><ul style='margin:2px 0 0 14px;padding:0'>{w}</ul></div></div>")


def legend(html_body, bottom, left=None, right=None):
    pos = f"bottom:{bottom}px;" + (f"left:{left}px;" if left is not None else f"right:{right}px;")
    return (f"<div style='position:fixed;{pos}z-index:9999;background:white;padding:8px 10px;border:1px solid #999;"
            f"border-radius:4px;font-size:11px;max-width:300px;box-shadow:0 1px 4px rgba(0,0,0,.3)'>{html_body}</div>")


def build_point_map(df, cfg, gdf, jjm_district, outpath, rep):
    if not HAS_FOLIUM:
        rep.warn("folium not installed - point map not generated.")
        return False
    d = df[df["coords_valid"]].copy()
    n_bad = int((~df["coords_valid"]).sum())
    n_out = int((d["coords_in_tn_bbox"] == False).sum())  # noqa: E712
    rep.p(f"Point map: {len(d)} plotted, {n_bad} excluded for invalid coordinates, "
          f"{n_out} plotted but outside the expected Tamil Nadu extent (flagged, not moved).")
    m = folium.Map(location=[10.9, 78.4], zoom_start=7, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="Base: OpenStreetMap").add_to(m)
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="OpenTopoMap (CC-BY-SA)", name="Base: topographic", show=False).add_to(m)
    if gdf is not None:
        folium.GeoJson(gdf[["geometry"]].assign(geometry=gdf.geometry.simplify(0.003)).to_json(), name="District boundaries",
                       style_function=lambda _: {"color": "#555", "weight": 1, "fillOpacity": 0}).add_to(m)
    smax = d["Priority_Score"].max()
    scale = lambda v: 4 + 8 * (0 if pd.isna(v) or not smax else v / smax)

    def layer(name, show, color_fn, size_fn, popup=False):
        fg = folium.FeatureGroup(name=name, show=show)
        for _, r in d.iterrows():
            folium.CircleMarker([r["latitude"], r["longitude"]], radius=size_fn(r), color="#333", weight=.5,
                                fill=True, fill_color=color_fn(r), fill_opacity=.8,
                                popup=folium.Popup(popup_html(r, cfg), max_width=380) if popup else None,
                                tooltip=f"{r['district']} | {r['Intervention_Category']}").add_to(fg)
        fg.add_to(m)

    layer("1. Priority (final score; green/yellow/red = Low/Medium/High)", True,
          lambda r: COLORS[r["Intervention_Category"]], lambda r: scale(r["Priority_Score"]), popup=True)
    layer("2. Mechanical failure risk (experimental model)", False,
          lambda r: ramp(r["Mechanical_Failure_Risk"], 0, d["Mechanical_Failure_Risk"].quantile(.95)), lambda r: 6)
    layer("3. Water-quality screening risk", False,
          lambda r: ramp(r["Water_Contamination_Risk"], 0, 1), lambda r: 6)
    layer("4. Groundwater monitoring context (station reading, NOT point location)", False,
          lambda r: ramp(-r["groundwater_level_m"] if r["gw_usable"] and r["groundwater_level_m"] < 0 else np.nan, 0,
                         max(1, -d.loc[d["gw_usable"] & (d["groundwater_level_m"] < 0), "groundwater_level_m"].min()
                             if (d["gw_usable"] & (d["groundwater_level_m"] < 0)).any() else 1)), lambda r: 6)
    layer("5. Climate vulnerability (simulated)", False,
          lambda r: ramp(r["Climate_Vulnerability_Index"], 0, 100), lambda r: 6)
    # JJM district layer: one marker per district centroid of mapped points, district-level value once
    fg = folium.FeatureGroup(name="6. JJM district tap coverage (district aggregate)", show=False)
    cent = d.groupby("district_key")[["latitude", "longitude"]].mean()
    for k, row in jjm_district.iterrows():
        if k not in cent.index:
            continue
        v = row["jjm_tap_coverage_pct"]
        col = "#888888" if pd.isna(v) else ramp(100 - v, 0, 60)  # low coverage = red
        folium.CircleMarker([cent.loc[k, "latitude"], cent.loc[k, "longitude"]], radius=11, color="#333", weight=1,
                            fill=True, fill_color=col, fill_opacity=.75,
                            tooltip=f"{row['district']}: JJM tap coverage {fmt(v, 1, '%')} (district-level; marker at mean of "
                                    "mapped points, not a district centroid)").add_to(fg)
    fg.add_to(m)
    # heatmap
    hp = d[d["Priority_Score"].notna()]
    if len(hp):
        w = (hp["Priority_Score"] - hp["Priority_Score"].min()) / (hp["Priority_Score"].max() - hp["Priority_Score"].min() + 1e-12)
        fgh = folium.FeatureGroup(name="Heatmap: priority-weighted density of mapped reference points", show=False)
        HeatMap(np.column_stack([hp["latitude"], hp["longitude"], w]).tolist(), radius=18, blur=22, min_opacity=.25).add_to(fgh)
        fgh.add_to(m)
    # cluster layer
    cl = d[d["Hotspot_Cluster_ID"].notna() & (d["Hotspot_Cluster_ID"] >= 0)]
    if len(cl):
        fgc = folium.FeatureGroup(name="Hotspot clusters (DBSCAN, haversine)", show=True)
        palette = ["#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#ff7f0e"]
        for cid, g in cl.groupby("Hotspot_Cluster_ID"):
            c = palette[int(cid) % len(palette)]
            folium.Circle([g["latitude"].mean(), g["longitude"].mean()],
                          radius=max(g.apply(lambda r: haversine_km(g["latitude"].mean(), g["longitude"].mean(),
                                                                       r["latitude"], r["longitude"]), axis=1).max(), 2) * 1000,
                          color=c, weight=2, fill=False, dash_array="6",
                          tooltip=f"Cluster #{int(cid)}: {len(g)} high-priority points (spatial concentration only; no causal claim)").add_to(fgc)
        fgc.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(legend(
        "<b>Priority category (Priority_Score, incl. climate adj.)</b><br>"
        f"<span style='color:{COLORS['Low']}'>●</span> Low &lt; {cfg._med:.3f}<br>"
        f"<span style='color:{COLORS['Medium']}'>●</span> Medium {cfg._med:.3f}–{cfg._hi:.3f}<br>"
        f"<span style='color:{COLORS['High']}'>●</span> High ≥ {cfg._hi:.3f}<br>"
        "<span style='color:#888'>●</span> Unrated/missing data<br>"
        "Other layers: green→yellow→red = low→high (groundwater: shallower→deeper; JJM: higher→lower coverage; grey = unavailable)<br>"
        "Marker size ∝ priority score.", 20, left=10)))
    m.get_root().html.add_child(folium.Element(legend(
        "<b>Read with care</b><br>Points are water-body / monitoring-station <b>reference proxies</b>, not verified pumps. "
        "Sensors, chemistry, population and status labels may be <b>synthetic</b>. JJM = district aggregate. "
        "Groundwater values belong to stations. Monsoon, RUL and 2031 values are <b>scenarios</b>. "
        f"{cfg.heat_note}", 20, right=10)))
    m.save(str(outpath))
    return True


# ----------------------------------------------------------------------------
# Objective 12 - choropleth
# ----------------------------------------------------------------------------
def district_summary(df, jjm_district, cfg):
    g = df.groupby("district_key")
    s = pd.DataFrame({
        "district": g["district"].first(),
        "mapped_locations": g.size(),
        "mean_Priority_Score": g["Priority_Score"].mean(),
        "high_priority_locations": g["Intervention_Category"].apply(lambda x: int((x == "High").sum())),
        "mean_Mechanical_Failure_Risk": g["Mechanical_Failure_Risk"].mean(),
        "mean_Water_Contamination_Risk": g["Water_Contamination_Risk"].mean(),
        "mean_WQI": g["WQI"].mean(),
        "mean_Climate_Vulnerability_Index": g["Climate_Vulnerability_Index"].mean(),
    })
    gwm = df[df["gw_usable"]].groupby("district_key")["groundwater_level_m"].agg(["mean", "count"])
    s["mean_groundwater_level_m"] = gwm["mean"].where(gwm["count"] >= 3)  # require >=3 usable readings
    s["groundwater_readings_used"] = gwm["count"]
    s["jjm_tap_coverage_pct"] = jjm_district["jjm_tap_coverage_pct"]   # district value ONCE per district
    return s


def build_choropleth(df, gdf, jjm_district, cfg, outpath, rep, metric="mean_Priority_Score"):
    if gdf is None or not HAS_FOLIUM:
        rep.p("Choropleth NOT produced: genuine district-boundary data unavailable (supply --boundary FILE).")
        rep.warn("District choropleth not produced (no valid boundary data).")
        return None
    s = district_summary(df, jjm_district, cfg)
    bkeys, dkeys = set(gdf["district_key"]), set(s.index)
    um_data, um_poly = sorted(dkeys - bkeys), sorted(bkeys - dkeys)
    rep.p(f"District join: {len(dkeys & bkeys)} matched; data districts without polygon: {um_data or 'none'}; "
          f"polygons without data: {um_poly or 'none'}")
    if um_data:
        rep.warn("Districts in data without polygon (not drawn, not merged into neighbours): " + ", ".join(
            s.loc[um_data, "district"]))
    g = gdf.merge(s.reset_index(), on="district_key", how="left")
    insufficient = g["mapped_locations"].fillna(0) < 3
    g.loc[insufficient, metric] = np.nan                       # insufficient support -> no-data
    vals = g[metric].dropna()
    m = folium.Map(location=[10.9, 78.4], zoom_start=7, tiles="OpenStreetMap", control_scale=True)
    if vals.empty:
        rep.warn("No district had enough data for the choropleth metric.")
        return None
    vmin, vmax = float(vals.min()), float(vals.max())
    import branca.colormap as bcm
    cmap = bcm.LinearColormap(["#2ca02c", "#f2c200", "#d62728"], vmin=vmin, vmax=vmax,
                              caption=f"{metric} (district mean of screening scores; grey = no data)")

    def style(feat):
        v = feat["properties"].get(metric)
        return {"fillColor": "#bdbdbd" if v is None or pd.isna(v) else cmap(v), "color": "#444",
                "weight": 1, "fillOpacity": .75}

    cols = ["district_key", "district", "mapped_locations", "mean_Priority_Score", "high_priority_locations",
            "mean_Mechanical_Failure_Risk", "mean_Water_Contamination_Risk", "mean_WQI",
            "mean_groundwater_level_m", "mean_Climate_Vulnerability_Index", "jjm_tap_coverage_pct"]
    gj = g[cols + ["geometry"]].copy()
    gj["geometry"] = gj.geometry.simplify(0.003)   # display only; analysis used full-resolution polygons
    gj["district"] = gj["district"].fillna(gj["district_key"].str.title())
    for c in cols[2:]:
        gj[c] = gj[c].round(3)
    gj_json = gj.to_json()
    aliases = ["District", "Mapped reference locations", "Mean priority score", "High-priority locations",
               "Mean mechanical failure risk (experimental)", "Mean water-contamination screening",
               "Mean WQI (synthetic chemistry)", "Mean groundwater level m (station-based, >=3 readings)",
               "Mean climate vulnerability (simulated)", "JJM tap coverage % (district-level)"]
    folium.GeoJson(gj_json, name="Districts", style_function=style,
                   highlight_function=lambda _: {"weight": 3, "color": "#000"},
                   tooltip=folium.GeoJsonTooltip(fields=cols[1:], aliases=aliases, localize=True, sticky=True,
                                                 labels=True, style="font-size:11px")).add_to(m)
    cmap.add_to(m)
    m.get_root().html.add_child(folium.Element(legend(
        "<b>District-level averages may conceal local hotspots.</b><br>Grey = no data / fewer than 3 mapped locations "
        "(not zero risk). Metrics derive from proxy locations and partly SYNTHETIC inputs; JJM is the source "
        "district value; groundwater readings are station-based; scores are screening indices, not probabilities.",
        20, left=10)))
    m.save(str(outpath))
    return s


# ----------------------------------------------------------------------------
# Objective 10 - output validation
# ----------------------------------------------------------------------------
REQUIRED_OUT = ["Mechanical_Failure_Risk", "Recovery_or_Maintenance_Risk", "Water_Contamination_Risk",
                "Priority_Score_Base", "Priority_Score", "Priority_Rank", "Intervention_Category",
                "Aquifer_RUL_Months", "Fluoride_RUL_Months", "TDS_RUL_Months", "Limiting_Chemical",
                "monsoon_deviation_pct", "Climate_Vulnerability_Index", "Climate_Adjustment_Factor",
                "Mechanical_Failure_Risk_2031", "Recovery_or_Maintenance_Risk_2031",
                "Water_Contamination_Risk_2031", "Priority_Score_2031", "Fluoride_2031", "TDS_2031",
                "WQI_2031", "CCI_2031", "chlorine_anomaly", "low_supply", "chlorine_data_missing",
                "lpcd_data_missing", "geospatial_data_quality"]


def validate_output(path, src_df, cfg, rep):
    checks = []
    back = pd.read_csv(path)
    checks.append(("output readable", True))
    checks.append(("required columns present", all(c in back.columns for c in REQUIRED_OUT)))
    for c in ["Mechanical_Failure_Risk", "Recovery_or_Maintenance_Risk", "Water_Contamination_Risk",
              "Mechanical_Failure_Risk_2031", "Recovery_or_Maintenance_Risk_2031", "Water_Contamination_Risk_2031"]:
        s = back[c].dropna()
        checks.append((f"{c} in [0,1]", bool(s.between(0, 1).all())))
    checks.append(("Aquifer_RUL_Months in [0,120]", bool(back["Aquifer_RUL_Months"].dropna().between(0, cfg.rul_cap_months).all())))
    checks.append(("row count == input", len(back) == len(src_df)))
    checks.append(("lat/lon unchanged", bool(np.allclose(back["latitude"].to_numpy(float), src_df["latitude"].to_numpy(float), equal_nan=True, atol=1e-9)
                                              and np.allclose(back["longitude"].to_numpy(float), src_df["longitude"].to_numpy(float), equal_nan=True, atol=1e-9))))
    for c in ["jjm_tap_coverage_pct", "jjm_total_households", "groundwater_level_m"]:
        if c in src_df:
            checks.append((f"{c} missing count preserved (no zero-fill)",
                           int(back[c].isna().sum()) == int(src_df[c].isna().sum())))
    cat = categorise(back["Priority_Score"], cfg._hi, cfg._med)
    checks.append(("categories match thresholds", bool((cat == back["Intervention_Category"]).all())))
    checks.append(("priority ranks consistent", bool(back.loc[back["Priority_Score"].notna(), "Priority_Rank"].min() == 1)))
    rep.head("OUTPUT VALIDATION")
    ok_all = True
    for name, ok in checks:
        rep.p(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        ok_all &= ok
    if not ok_all:
        rep.warn("One or more output validation checks FAILED.")
    return ok_all


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run(args) -> int:
    rep = Report()
    cfg = Config(seed=args.seed, cluster_radius_km=args.cluster_radius_km,
                 cluster_min_samples=args.cluster_min_samples, cluster_rule=args.cluster_rule,
                 climate_max_adjustment=args.climate_max_adjustment)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rep.head("1. LOAD & PREPARE")
    raw = load_data(Path(args.input), rep)
    validate_inputs(raw, rep)
    sensors = resolve_sensor_mapping(raw, parse_sensor_map(args.sensor_map), args.allow_unverified_sensors, rep)
    rep.p("Sensor mapping used: " + str(sensors["map"]) + ("  [UNVERIFIED: " + ", ".join(sensors["unverified"]) + "]"
                                                           if sensors["unverified"] else "  [verified by name]"))
    df = prepare_frame(raw, sensors, cfg, rep)
    df["sensor_mapping_status"] = "UNVERIFIED (user-asserted)" if sensors["unverified"] else "verified by column name"
    for c in CANONICAL_FEATURES:
        df[f"{c}_missing_in_source"] = df[c].isna()
    df["model_inputs_imputed"] = df[CANONICAL_FEATURES].isna().any(axis=1)

    # WQI formula frozen on BASELINE data; confirm it reproduces the file's WQI
    chem = df[["Fluoride_Level", "Nitrate_Level", "TDS"]].astype(float)
    wqi_model = WQIModel(chem.min().to_numpy(), chem.max().to_numpy())
    recomputed = wqi_model(chem["Fluoride_Level"], chem["Nitrate_Level"], chem["TDS"])
    diff = np.nanmax(np.abs(recomputed - df["WQI"].to_numpy(float))) if df["WQI"].notna().any() else np.nan
    rep.p(f"WQI formula reproduction check (max |recomputed - file WQI|): {diff:.3f}")
    if diff > 0.05:
        rep.warn(f"Recomputed baseline WQI deviates from file WQI by up to {diff:.2f}; 2031 WQI may not be "
                 "strictly comparable with baseline WQI.")
    cci_re = (df["TDS"] / 1000) * (df["motor_temp"] / 40)
    cdiff = np.nanmax(np.abs(cci_re - df["CCI"]))
    rep.p(f"CCI formula reproduction check (max abs diff): {cdiff:.4f}")
    if cdiff > 0.001:
        rep.warn(f"CCI formula (TDS/1000 * motor_temp/40) deviates from file CCI by up to {cdiff:.4f}.")

    rep.head("2. MODEL TRAINING (imbalance-aware XGBoost)")
    rep.p(f"XGBClassifier(multi:softprob, mlogloss, n_estimators={cfg.n_estimators}, max_depth={cfg.max_depth}, "
          f"learning_rate={cfg.learning_rate}, random_state={cfg.seed}); sample_weight = inverse class frequency")
    model, imputer, metrics = train_and_evaluate(df, cfg, rep)
    print_metrics(metrics, rep)

    rep.head("3-5. BASELINE SCORES")
    P = predict_probs(model, imputer, df)
    df["Mechanical_Failure_Risk"] = P[:, 2]
    df["Recovery_or_Maintenance_Risk"] = P[:, 1]
    df["Normal_Class_Probability"] = P[:, 0]
    df["model_status"] = "EXPERIMENTAL (synthetic labels); not a validated failure probability"
    df["Water_Contamination_Risk"] = contamination_score(df["Fluoride_Level"], df["Nitrate_Level"], df["TDS"], df["WQI"], cfg)
    df["Water_Contamination_Risk_available"] = df["Water_Contamination_Risk"].notna()
    df["Water_Contamination_Risk_note"] = "Screening index 0-1 (threshold-normalised); NOT a calibrated probability"
    df = constraint_flags(df, cfg)
    df["Priority_Score_Base"] = priority_base(df["Mechanical_Failure_Risk"], df["Water_Contamination_Risk"],
                                              df["CCI"], df["population_served"])
    df["Priority_Score_Penalised"], df["Penalty_Multiplier"] = apply_penalties(
        df["Priority_Score_Base"], df["chlorine_anomaly"], df["low_supply"], cfg)
    df["Priority_Score_Incomplete"] = df["Priority_Score_Base"].isna()
    df = rul_block(df, cfg)
    df = climate_block(df, cfg, cfg.seed + 7)
    df["Priority_Score"] = df["Priority_Score_Penalised"] * df["Climate_Adjustment_Factor"]
    # Categories: percentile cut-points computed ONCE on the final baseline score and reused everywhere
    cfg._hi = float(df["Priority_Score"].quantile(cfg.high_priority_quantile))
    cfg._med = float(df["Priority_Score"].quantile(cfg.medium_priority_quantile))
    df["Intervention_Category"] = categorise(df["Priority_Score"], cfg._hi, cfg._med)
    df["Priority_Rank"] = df["Priority_Score"].rank(ascending=False, method="min")
    df["Priority_note"] = ("Screening score (not a probability). Final Priority_Score = Base x penalties x climate factor. "
                           f"Categories: High >= {cfg._hi:.4f} (top {int((1 - cfg.high_priority_quantile) * 100)}%), "
                           f"Medium >= {cfg._med:.4f}, else Low")
    rep.p(f"Category cut-points (percentile based): Medium >= {cfg._med:.4f}, High >= {cfg._hi:.4f}")
    rep.p(f"Rows with incomplete priority inputs (Unrated): {int(df['Priority_Score_Incomplete'].sum())}")

    rep.head("8. 2031 SCENARIO")
    fut = future_block(df, cfg, model, imputer, wqi_model, cfg._hi, cfg._med, rep)
    df = pd.concat([df, fut], axis=1)
    df["scenario_label"] = SCENARIO_NOTE

    rep.head("13. HOTSPOT CLUSTERING")
    df, sizes = run_clustering(df, cfg, rep)

    rep.head("GEOGRAPHIC VALIDATION")
    boundary_path = find_boundary(args.boundary, [Path.cwd(), Path(args.input).resolve().parent, Path(__file__).resolve().parent])
    gdf, _ = (load_boundary(boundary_path, rep) if boundary_path else (None, None))
    if boundary_path is None:
        rep.p("No district-boundary file found (use --boundary FILE.geojson|.shp|.kml).")
    df = boundary_point_checks(df, gdf, rep)
    n_out_bbox = int((df["coords_valid"] & ~df["coords_in_tn_bbox"]).sum())
    rep.p(f"Valid coordinates: {int(df['coords_valid'].sum())}/{len(df)}; outside TN bbox: {n_out_bbox}")

    # JJM district-level table: one value per district
    jjm = df.groupby("district_key").agg(district=("district", "first"),
                                         jjm_tap_coverage_pct=("jjm_tap_coverage_pct", "first"))
    jjm_conflict = df.groupby("district_key")["jjm_tap_coverage_pct"].nunique(dropna=False)
    if (jjm_conflict > 1).any():
        rep.warn("JJM coverage varies within a district - expected a single district-level value.")

    # ---- write dataset ----
    df["Priority_Rank"] = df["Priority_Rank"].astype("Int64")
    df["Hotspot_Cluster_ID"] = df["Hotspot_Cluster_ID"].astype("Int64")
    src_cols = list(raw.columns)
    new_cols = [c for c in df.columns if c not in src_cols]
    out = df[src_cols + new_cols].copy()
    out_path = outdir / "predictive_output.csv"
    out.to_csv(out_path, index=False)

    # ---- clusters CSV ----
    cl_cols = ["latitude", "longitude", "district", "Priority_Score", "Intervention_Category",
               "groundwater_station", "groundwater_level_m", "groundwater_observation_time",
               "groundwater_station_distance_km", "jjm_tap_coverage_pct", "Hotspot_Cluster_ID",
               "Hotspot_Cluster_Label", "geospatial_data_quality"]
    cl_path = outdir / "tamilnadu_hotspot_clusters.csv"
    df[select_cluster_points(df, cfg)][cl_cols].to_csv(cl_path, index=False)

    # ---- maps ----
    rep.head("MAPS")
    pt_path = outdir / "tamilnadu_water_hotspots.html"
    pt_ok = build_point_map(df, cfg, gdf, jjm, pt_path, rep)
    ch_path = outdir / "tamilnadu_district_risk_map.html"
    summ = build_choropleth(df, gdf, jjm, cfg, ch_path, rep)
    if summ is not None:
        summ.to_csv(outdir / "district_summary.csv")

    # ---- validation + report ----
    ok = validate_output(out_path, raw, cfg, rep)
    rep.head("SUMMARY")
    rep.p(f"Rows processed: {len(df)}   rows with valid coordinates: {int(df['coords_valid'].sum())}")
    rep.p("District distribution:\n" + df["district"].value_counts().to_string())
    rep.p("Machine-status distribution:\n" + df["status_group"].map(STATUS_NAMES).value_counts().to_string())
    rep.p(f"Model: macro-F1={metrics['macro_f1']:.3f}  balanced-acc={metrics['balanced_accuracy']:.3f} "
          "(EXPERIMENTAL; see notes above)" if not np.isnan(metrics["macro_f1"]) else "Model metrics unavailable")
    rep.p(f"High-priority locations: {int((df['Intervention_Category'] == 'High').sum())}")
    miss = out[[c for c in src_cols if out[c].isna().any()]].isna().sum()
    rep.p("Missing-value summary (source columns with gaps):\n" + miss.to_string())
    gwbad = {"sentinel": int(df["gw_sentinel_value"].sum()), "implausible": int(df["gw_implausible_value"].sum()),
             "not recent": int((~df["gw_obs_recent"]).sum()), "station too far": int((~df["gw_station_near"]).sum()),
             "usable": int(df["gw_usable"].sum())}
    rep.p(f"Groundwater readings: {gwbad}")
    if gwbad["sentinel"]:
        rep.warn(f"{gwbad['sentinel']} groundwater values are no-data sentinels (e.g. -999.999); excluded from "
                 "stress/means but kept unchanged in the CSV.")
    if df["sensor_values_synthetic"].any():
        rep.warn("Sensor values / machine-status labels are SYNTHETIC; model is experimental.")
    if df["chemistry_values_synthetic"].any():
        rep.warn("Water chemistry is SYNTHETIC; contamination, WQI, RUL and 2031 outputs are illustrative.")
    if (df["geospatial_data_quality"] != "verified_asset").all():
        rep.warn("No coordinate is a verified pump; all points are proxies/unknown provenance.")
    rep.warn("Monsoon deviation, RUL and 2031 columns are scenarios, not forecasts. Score = screening index.")
    rep.p("\nData-quality warnings:")
    for w in rep.warnings:
        rep.p("  - " + w)
    rep.p("\nOutput files:")
    for p, made in [(out_path, True), (pt_path, pt_ok), (ch_path, summ is not None), (cl_path, True)]:
        rep.p(f"  {'[ok]     ' if made else '[skipped]'} {p}")
    (outdir / "validation_report.txt").write_text("\n".join(rep.lines), encoding="utf-8")
    return 0 if ok else 2


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Tamil Nadu water-infrastructure predictive engine (prototype).")
    ap.add_argument("input", nargs="?", default="fused_tn_pumps.csv", help="fused CSV from data_fusion.py")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="random seed (model + monsoon scenario)")
    ap.add_argument("--sensor-map", nargs="*", default=[], metavar="CANON=COLUMN",
                    help="verified sensor mapping, e.g. vibration=sensor_02 flow_rate=sensor_04")
    ap.add_argument("--allow-unverified-sensors", action="store_true",
                    help="accept user-asserted sensor mappings that cannot be verified from the file (flagged UNVERIFIED)")
    ap.add_argument("--boundary", default=None, help="district boundary file (.geojson/.shp/.kml)")
    ap.add_argument("--cluster-radius-km", type=float, default=25.0)
    ap.add_argument("--cluster-min-samples", type=int, default=5)
    ap.add_argument("--cluster-rule", default="High", help="'High' or 'quantile:0.9'")
    ap.add_argument("--climate-max-adjustment", type=float, default=0.25)
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
