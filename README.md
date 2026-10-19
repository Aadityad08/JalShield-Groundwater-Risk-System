# JalShield: Water Infrastructure Intelligence for Tamil Nadu

JalShield is a decision-support prototype that ranks water-supply locations in Tamil Nadu by how urgently they need maintenance or water-quality intervention, then allocates a limited repair budget across them. It links an ESP32 sensing node, a data-fusion step, an XGBoost-based analytics engine and a Streamlit dashboard.

> **Prototype status.** Pump-sensor values, water chemistry, population, chlorine readings and machine-status labels in the bundled dataset are **synthetic**. Locations are **proxies** (water-body or groundwater-station coordinates), not surveyed pumps. All scores are screening indices, not calibrated probabilities, and the 2031 outputs are scenarios, not forecasts. See [Data provenance](#data-provenance).

---

## Features

- **ESP32 telemetry node:** reads acceleration (MPU6050), water temperature (DS18B20) and an analog TDS probe, shows them on an SSD1306 OLED, and sends one JSON line every 2 s over USB serial.
- **Data fusion:** combines pump-sensor records with Tamil Nadu groundwater-station readings, Jal Dharohar water-body coordinates and Jal Jeevan Mission (JJM) tap-coverage figures for all 38 districts.
- **Failure model:** imbalance-aware XGBoost classifier (NORMAL / RECOVERING / BROKEN) using inverse-frequency sample weights and a stratified hold-out evaluation.
- **Contamination index:** threshold-normalised score from fluoride, nitrate, TDS and WQI.
- **Priority Score:** combines failure risk, contamination risk, corrosion index (CCI), population served, chlorine and low-supply penalties, and a climate adjustment. Locations are categorised High / Medium / Low by percentile.
- **Scenarios:** a 2031 projection (fluoride +5%/yr, TDS +8%/yr), a chemical-threshold remaining-useful-life (RUL) indicator, and a climate vulnerability index (CVI).
- **Hotspot clustering:** DBSCAN with the haversine metric (25 km radius, minimum 5 points) on high-priority locations.
- **Greedy dispatch:** picks locations by value for money until the crew or budget limit is reached, and exports a CSV manifest.
- **Dashboard:** Streamlit app with six pages and Folium maps (risk markers, priority heat, groundwater level, projected 2031).
- **Provenance and validation:** every output row carries synthetic-data and location-quality flags, and the engine runs 16 output checks and writes a validation report.

---

## Repository structure

```
JalShield/
├── firmware/
│   └── esp32_jalshield_usb.ino     # ESP32 firmware: sensors, OLED, JSON over USB
├── software/
│   ├── app.py                      # Streamlit dashboard (six pages, greedy dispatch)
│   ├── data_layer.py               # Data loading and Folium map helpers
│   ├── data_fusion.py              # Builds fused_tn_pumps.csv from sensor + reference data
│   ├── predictive_engine.py        # XGBoost, scores, scenarios, clustering, maps, validation
│   └── fused_tn_pumps.csv          # Fused dataset (1,000 rows x 85 columns)
└── README.md
```

`app.py` currently defines its own copies of the helpers in `data_layer.py`.

---

## How it works

```
Reference data ─┐
(KML, groundwater,├─> data_fusion.py ─> fused_tn_pumps.csv ─> predictive_engine.py ─> predictive_output.csv ─> app.py
 JJM table)      │                                           └─> tamilnadu_hotspot_clusters.csv, maps, report
ESP32 node ── JSON over USB ───────────────────────────────────────────────────────────────────────────────┘ (Live Sensors page)
```

The ESP32 stream is captured on demand on the Live Sensors page. It does not feed the batch model.

### Scoring

Water quality and corrosion indicators (from `data_fusion.py`):

```
pollution = 0.40*F' + 0.35*N' + 0.25*TDS'          (min-max scaled)
WQI       = clip(100*(1 - pollution) - 20*[F>1.5] - 20*[N>45], 0, 100)
CCI       = (TDS / 1000) * (motor_temp / 40)
```

Contamination risk and priority (from `predictive_engine.py`):

```
WCR      = 0.35*min(F/1.5,1) + 0.25*min(N/45,1) + 0.20*min(TDS/2000,1) + 0.20*clip((100-WQI)/50, 0, 1)
Priority = (0.55*MFR + 0.25*WCR) * (0.20*CCI) * ln(1 + population)
           * (1 + 0.15*chlorine_anomaly + 0.15*low_supply) * (1 + 0.25*CVI/100)
```

`MFR` is the model's predicted BROKEN probability. A chlorine anomaly means residual chlorine outside 0.2 to 0.5, and low supply means below 55 lpcd. Categories: top 20% High, next 30% Medium, rest Low. The dashboard labels this score "PIS".

### Greedy dispatch

```
cost       = 120,000 * Priority + 3 * population
efficiency = population * Priority / cost
```

Candidates are sorted by efficiency and accepted while the number of crews and the running cost stay within the limits you set on the dashboard. The cost model is an assumption of the prototype.

---

## Hardware

| Function | Part | Connection |
| --- | --- | --- |
| Acceleration (vibration proxy) | MPU6050, range ±8 g | I2C, SDA 21, SCL 22 |
| Water temperature | DS18B20 | OneWire, GPIO 4 |
| Total dissolved solids | Analog TDS probe | ADC, GPIO 32 (12-bit) |
| Display | SSD1306 OLED 128x64 | I2C, address 0x3C |
| Flow | **Simulated** (fixed 90 L/min) | Not yet connected |

Arduino libraries: Adafruit MPU6050, Adafruit Unified Sensor, Adafruit GFX, Adafruit SSD1306, OneWire, DallasTemperature.

Serial output (115200 baud, one object per line every 2 s):

```json
{"vibration_ms2":9.81,"flow_lpm":90.0,"tds_ppm":0.0,"temp_c":25.00}
```

Notes: TDS is a placeholder voltage-to-ppm conversion with no calibration or temperature compensation. Vibration is the total acceleration magnitude, which includes gravity.

---

## Setup

### Requirements

```
pip install pandas numpy scikit-learn xgboost streamlit folium streamlit-folium pyserial
pip install geopandas   # optional: district boundary checks and choropleth
```

### Run

From the `software/` folder:

```
# 1. (Optional) rebuild the fused dataset. A ready-made copy is included.
python data_fusion.py --reference-zip Datasets.zip

# 2. Score the data and write the outputs next to app.py
python predictive_engine.py fused_tn_pumps.csv \
    --sensor-map vibration=sensor_02 flow_rate=sensor_04 \
    --allow-unverified-sensors --outdir .

# 3. Start the dashboard
streamlit run app.py
```

- `data_fusion.py` generates a synthetic sensor file if `sensor.csv` is absent. The reference archive (KML water bodies, groundwater CSVs) is optional; without it, locations and groundwater columns are left empty.
- The engine refuses to guess what a generic sensor column measures. You must name the vibration and flow columns with `--sensor-map`, and confirm unverified mappings with `--allow-unverified-sensors`. They are recorded as unverified in the output.
- Add `--boundary districts.geojson` (or `.shp` / `.kml`) to enable boundary validation and the district choropleth.
- Other options: `--seed`, `--cluster-radius-km`, `--cluster-min-samples`, `--cluster-rule`, `--climate-max-adjustment`.

### Outputs

| File | Contents |
| --- | --- |
| `predictive_output.csv` | All input columns plus risks, scores, categories, scenarios, RUL, CVI and quality flags |
| `tamilnadu_hotspot_clusters.csv` | High-priority locations with DBSCAN cluster labels |
| `tamilnadu_water_hotspots.html` | Interactive point map |
| `tamilnadu_district_risk_map.html` | District choropleth (only with a boundary file) |
| `validation_report.txt` | Console report with data-quality warnings |

### Live sensors

Flash `firmware/esp32_jalshield_usb.ino`, connect the ESP32 by USB, close the Arduino Serial Monitor (only one program can use the port), open the **Live Sensors** page, choose the port and press the read button. The latest packet is saved to `live_hardware_data.json`.

---

## Dashboard pages

| Page | Purpose |
| --- | --- |
| Overview | Headline counts, map and priority list |
| Predictions | 2031 scenario view |
| Water Quality | District filter and sortable quality table |
| Maintenance & Dispatch | Budget and crew sliders, dispatch manifest, CSV downloads |
| Live Sensors | Manual USB capture from the ESP32 |
| Data Explorer | Column search and dataset preview |

If no data file is found, the app opens in a preview mode with illustrative examples.

---

## Data provenance

| Data | Source | Status |
| --- | --- | --- |
| Pump sensor channels and `machine_status` | Generated by `data_fusion.py` | Synthetic |
| Fluoride, nitrate, TDS, residual chlorine, population | Random draws in `data_fusion.py` | Synthetic |
| Latitude and longitude | Jal Dharohar water-body points, or groundwater-station fallback | Real proxy, not pump locations |
| Groundwater level | Tamil Nadu groundwater-level station data (latest reading) | Real, from the nearest station |
| Households and tap coverage | JJM FHTC district table | Real, district-level (Chennai missing) |

Replace the synthetic fields with real telemetry and laboratory or CGWB observations before using any score operationally.

---

## Limitations

- The 2031 outputs, RUL and monsoon deviation are scenarios, not forecasts.
- Groundwater readings come from stations up to 15 km away, and the depth sign convention is assumed.
- Priority weights, penalties and the dispatch cost model are design choices, not calibrated values.
- Greedy selection is not guaranteed to be optimal.
- TDS is uncalibrated and flow is simulated on the current node.

## Project context

Engineering capstone / final year project. [Add authors, guide, institution and licence.]
