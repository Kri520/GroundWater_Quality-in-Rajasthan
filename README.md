# Groundwater Quality ML — Rajasthan

Machine-learning pipeline that predicts **drinking-water quality (BIS 10500:2012 class)** and **irrigation sodicity (SAR)** for groundwater wells in Rajasthan, India, using only field-measurable inputs (pH, EC, GPS location, year). Built on 60+ years (1961–2025) of manual chemical-parameter monitoring data from the **Central Ground Water Board (CGWB)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kri520/GroundWater_Quality-in-Rajasthan/blob/main/Groundwater_Quality_ML_Rajasthan.ipynb)

---

## Why this project

A full chemical panel (Ca, Mg, Na, Cl, SO4, alkalinity, hardness, etc.) requires a lab. A field officer with a ₹2,000 handheld meter only has **pH and EC**. This notebook asks: *how much water-quality risk can we screen for with just a field kit and a GPS pin, without leaking the lab-only ions into the model?*

## Data source

- **CGWB manual groundwater quality monitoring**, Rajasthan, 1961–2025 (`gwq_chemical_parameter_manual_cgwb_rj_1961_2025.csv`)
- Standards used: **BIS 10500:2012** (drinking water), **USSL / Richards 1954** (irrigation sodicity), **APHA 1030E** (analytical QA)
- The raw CSV is **not included** in this repo — place it alongside the notebook (or update `CSV_PATH` in the "Load the raw CGWB file" cell) before running.

## Pipeline overview

| Stage | What happens |
|---|---|
| **1. Environment setup** | Imports, reproducible seeds, output folders (`outputs/figures`, `outputs/models`, `outputs/tables`) |
| **2. Load raw data** | Reads the CGWB CSV, renames verbose column headers to compact chemistry symbols (pH, EC, TDS, Cl, SO4, TH, Ca, Mg, Na, Fe, …) |
| **3. Data quality audit** | Drops fully-empty columns; investigates and drops **TDS** (near-perfectly derived from EC — would leak into the target); flags physically implausible values against known valid ranges; recovers missing alkalinity stoichiometrically from HCO3/CO3; removes duplicate (station, timestamp) records |
| **4. Analytical validation** | Computes **ionic Charge Balance Error (CBE)** in meq/L and rejects samples with CBE > 10% — an electroneutrality check that filters out lab/transcription errors |
| **5. Hydrochemical feature engineering** | Derives classical irrigation-suitability indices: **SAR** (Sodium Adsorption Ratio), **%Na** (Wilcox), **RSC** (Residual Sodium Carbonate, Eaton 1950), ionic strength, and geographic/geo-spatial features |
| **6. Target construction** | Builds the **Water Quality Index (WQI)** from BIS 10500:2012 limits with expert-assigned weights (TDS/EC deliberately excluded so it stays a clean predictor, not a label leak); bins WQI into quality classes; derives **USSL salinity–sodicity classes** as a secondary irrigation target |
| **7. Exploratory data analysis** | Univariate distributions, Spearman correlation heatmap, Theil–Sen trend / Mann–Kendall test over time, district-level "% unsafe" burden, spatial salinity and quality maps |
| **8. Feature-set ablation** | Compares geography-only vs. +spatial k-NN vs. +field kit (pH, EC) feature blocks to quantify how much a cheap field meter is actually worth |
| **9. Model zoo & tuning** | Logistic Regression, KNN, Random Forest, Extra Trees, HistGradientBoosting (tuned via `RandomizedSearchCV`) under **grouped, stratified cross-validation** (`StratifiedGroupKFold` on well/station — so no well leaks between train and test) |
| **10. Evaluation** | Classification report, confusion matrices, ordinal ±1-class error, quadratic-weighted kappa, binary "unsafe well" screening recall/precision, probability calibration curves |
| **11. Explainability** | Permutation importance, partial-dependence plots (incl. pH × EC interaction surface), SHAP summary plot (if installed) |
| **12. Temporal hold-out** | A stricter test: train on ≤2016, predict on >2016, to simulate genuinely forecasting a well's *next* visit |
| **13. SAR regression** | Log-space HistGradientBoosting regressor with **Conformalized Quantile Regression** for calibrated 80% prediction intervals on irrigation sodicity, translated back into USSL hazard classes |
| **14. Error mapping** | Spatial plot of where classification errors concentrate |
| **15. Deployment artefact** | Saves a self-contained `joblib` model bundle (model + feature list + BIS standards + k-NN reference table) and a `predict_water_quality(pH, EC, lat, lon, year)` inference function, demoed on real wells and round-trip tested after reload |
| **16. Results summary** | Final metrics table for all three modelling tasks, exported to `outputs/tables/final_summary.csv` |

## Key design choices

- **No label leakage**: every ion that builds the WQI (TH, Cl, SO4, ALK, Ca, Mg, Na, Fe, CO3, HCO3, CBE, USSL) is explicitly banned from the feature set — the model only ever sees what a field kit + GPS can provide.
- **Grouped CV, not random CV**: splits are stratified by *well station*, so the same well never appears in both train and test — this is what makes the reported accuracy trustworthy for "unseen well" predictions.
- **Two realistic deployment scenarios**: (a) a brand-new, never-visited well, and (b) a monitored well's *next* visit (temporal hold-out), which are evaluated separately.
- **Safety-critical framing**: alongside multi-class accuracy, the notebook reports the binary "is this well unsafe (Very Poor / Unfit)?" recall/precision, since missing an unsafe well is the costlier error.

## Outputs

Running the notebook creates:

```
outputs/
├── figures/    # all plots (missingness, correlation, EDA, model comparison, SHAP, PDP, error maps, ...)
├── tables/     # district_summary.csv, permutation_importance.csv, final_summary.csv
└── models/     # gwq_rajasthan_wqi_classifier.joblib  (deployable model bundle)
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
joblib
```

Optional (auto-detected, used if present — the notebook degrades gracefully without them):

```
shap        # per-sample explanations
xgboost / lightgbm / catboost
optuna
folium
imblearn
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib shap
```

## How to run

1. Clone this repo and place the CGWB CSV (`gwq_chemical_parameter_manual_cgwb_rj_1961_2025.csv`) in the repo root, **or** open the notebook in Google Colab via the badge above and upload the CSV to `/content/`.
2. Run all cells top to bottom — the notebook is fully reproducible (`RANDOM_STATE = 42`).
3. The trained model bundle and all figures/tables are written to `outputs/` on completion.

## Using the trained model

```python
import joblib
artefact = joblib.load("outputs/models/gwq_rajasthan_wqi_classifier.joblib")

result = predict_water_quality(
    pH=7.8, EC=620,              # field-meter readings
    latitude=26.90, longitude=75.80,  # GPS
    year=2025,
    artefact=artefact
)
print(result["predicted_class"], result["confidence"])
```

## Repository structure

```
.
├── Groundwater_Quality_ML_Rajasthan.ipynb   # main notebook (this pipeline)
├── gwq_chemical_parameter_manual_cgwb_rj_1961_2025.csv                               # generated on run (figures, tables, model)
└── README.md
```

## Credits

- **Data:** Central Ground Water Board (CGWB), Ministry of Jal Shakti, Government of India
- **Standards:** BIS 10500:2012 (drinking water), USSL / Richards (1954), APHA 1030E
- **Methods referenced:** Eaton (1950) RSC, Wilcox %Na, Theil–Sen / Mann–Kendall trend tests, Conformalized Quantile Regression

