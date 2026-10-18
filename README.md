# JalShield: Cyber-Physical Water Risk Forecasting & Resource Allocation System

JalShield is an end-to-end cyber-physical architecture designed to automate groundwater quality diagnostics, evaluate multi-district vulnerabilities, and execute automated resource allocation. The project seamlessly integrates a hardware sensing layer with a machine learning predictive core and an algorithmic optimization backend.

---

##  Key System Features
*   **Edge Hardware Ingestion:** Real-time data logging of Total Dissolved Solids (TDS), fluid temperature, and flow velocity via an ESP32 microcontroller core.
*   **High-Level Data Fusion:** Disparate raw sensor streams are continuously fused with static, local ground-truth water tables to remove telemetry noise and sensor spikes.
*   **Priority Intervention Score (PIS):** A multi-criteria mathematical formula that aggregates normalized risk parameters through a weighted arithmetic sum to identify systemic weak points.
*   **XGBoost Machine Learning Regression:** Advanced gradient-boosted tree learning used to model complex, non-linear contamination parameters without human bias.
*   **Future Prediction Analysis:** Time-series trend extrapolation tracking risk trajectories over proactive 24-to-72-hour operational windows.
*   **Greedy Resource Allocation:** An automated optimization engine that processes nodes and distributes finite utility resources greedily based on peak live and forecasted risks.
*   **Dual GIS Hotspot Dashboard:** An interactive web portal running a 2000 ms telemetry refresh tick, visualizing active groundwater hotspots alongside future projected vulnerability areas.

---

##  Project Repository Structure

```text
JalShield/
├── firmware/
│   └── esp32_jalshield_usb.ino      # ESP32 C++ Telemetry & LED Sampling Firmware
├── software/
│   ├── app.py                       # Dash Visual Web Engine & Dual Mapping Layout
│   ├── data_layer.py                # Serial Listener Port Ingestion Daemon
│   ├── data_fusion.py               # Live-to-Static Feature Aggregator Module
│   ├── predictive_engine.py         # XGBoost Modeler, PIS Core, & Greedy Optimizer
│   └── fused_tn_pumps.csv           # Regional Hydro-Environmental Groundwater Dataset
└── README.md                        # Project Documentation
```

---

## System Mathematical Framework

### 1. Priority Intervention Score (PIS) Formula
Raw sensor dimensions are mapped to dimensionless hazard sub-indicators (\(R_i \in [1, 10]\)) and calculated via a relative weighted sum:

\[\mathbf{PIS} = \sum_{i=1}^{4} (W_i \times R_i)\]

\[\text{Subject to constraints: } \sum_{i=1}^{4} W_i = 1.0\]

### 2. Greedy Allocation Optimization Logic
The target node selection space is evaluated sequentially based on immediate local vulnerability spikes. The allocator greedily targets resources to nodes displaying the highest combined active and forecasted risk profiles:

\[\text{Target} = \max \left( \mathbf{PIS}_{\text{Current}} + \Delta \mathbf{PIS}_{\text{Forecast}} \right)\]

---

## Software Setup & Local Execution

### Prerequisites
Ensure your local Python ecosystem has the necessary structural libraries installed:
```bash
pip install xgboost pandas numpy dash dash-bootstrap-components scikit-learn
```

### Execution Pipeline
To initiate the internal validation pipeline, construct the dataset models, and launch the asynchronous real-time user interface, run the entry point script:
```bash
python software/app.py
```
Once the machine learning verification steps report stability parameters in your terminal console, navigate to the local network port in your browser:
```text
http://127.0.0
```

---
## Data Sourcing & Fusion Acknowledgments
The baseline historical datasets and regional groundwater thresholds mapped inside this application are fused using field statistics and compiled environmental records. 

*   **Primary Processing Data:** `fused_tn_pumps.csv` (Locally maintained core lookup metrics).
*   **External Hydrological Baselines:** Regional regional telemetry models sourced from public water records.

## 👥 Project Development & Academic Context
*   **Project Name:** JalShield Water Management Platform
*   **Course Context:** Engineering Capstone / Final Year Project Submission
*   **Hardware Interface Core:** ESP32 Breadboard Node Assembly
*   **Analytics Scope:** Tabular Multi-Sensor Regression & Discrete Greedy Allocation Optimization
