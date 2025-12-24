# SAFAL — System for Air Quality Forecasting (Research Archive)

**Short:** SAFAL is a research-grade data-fusion and machine learning pipeline for short-term air quality forecasting. The primary targets are **NO₂ and O₃**, with **PM2.5** used for validated baseline experiments.  
This repository is an **archival snapshot** of the project prepared after the **Smart India Hackathon 2025 (Top-6 worldwide)**.  
It contains data-processing pipelines, physics-based feature engineering, model artifacts, and documentation — **not** a production web application.

---

## Highlights (quick facts)
- Final engineered training table (reported): **102,720 rows × 83 columns**  
- INSAT processing: **~1,039,390** image crops processed; **228,973** cached INSAT-derived features  
- PM2.5 baseline (XGBoost): **RMSE ≈ 27.98 ± 5.34 μg/m³**, **R² ≈ 0.676 ± 0.154**  
- Models / targets: **NO₂, O₃ (primary)**; PM2.5 (baseline & validation experiments)  
- Context: **Top-6 worldwide**, Smart India Hackathon 2025  
- Role: **Lead Data Scientist / ML Engineer** — data acquisition, feature engineering, modeling, validation  

---

## Repository structure
SAFAL-aq-forecasting/
├── inference_engine/ # inference logic, demo scripts, evaluation, model artifacts
├── data_pipeline/ # feature engineering and dataset construction scripts
├── ingestion/ # ERA5, MERRA2, satellite ingestion pipelines
├── station_metadata/ # station coordinates and metadata
├── docs/ # reports, figures, submission material
├── legacy/ # deprecated / exploratory experiments
├── DATA_NOTICE.md # data availability & reproduction notes
├── TECHNICAL_REPORT.md # detailed technical report (canonical reference)
└── README.md, LICENSE, .gitignore


---

## Quickstart (lightweight, reproducible workflow)

> **Note:** This repository intentionally does **not** include large raw satellite or reanalysis datasets.  
> See `DATA_NOTICE.md` for details on obtaining raw inputs and reproducing the full pipeline.

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/SAFAL-aq-forecasting.git
cd SAFAL-aq-forecasting

2. (Optional) Create a Python virtual environment

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

3. Install minimal dependencies (placeholders)
pip install -r inference_engine/requirements.txt
pip install -r data_pipeline/requirements.txt

4. (Optional) Create a small sample dataset & demo model
If scripts/make_sample.py exists:
python scripts/make_sample.py \
  --input "path/to/full_final_engineered_dataset.parquet" \
  --rows 1000 \
  --target NO2_target

This generates:

a small sample dataset under data_pipeline/

a lightweight demo model under inference_engine/models/

5. (Optional) Run demo inference

If demo scripts are present:

python inference_engine/demo/run_demo.py \
  --sample data_pipeline/sample_small.parquet

If not, inspect inference_engine/inference/ and evaluation/ for model loading and scoring examples.

Data & reproducibility

Large raw datasets (INSAT, ERA5, MERRA2, full engineered tables) are excluded to conserve space and respect licensing.

DATA_NOTICE.md explains original data sources and how to re-run the ingestion pipelines.

The repository contains all scripts and logic required to reproduce results once raw data are supplied following the documented structure.

How to read this project (for reviewers)

Start with TECHNICAL_REPORT.md — this is the canonical technical explanation.

Review data_pipeline/feature_engineering/ for physics-based features:

Ventilation Coefficient (BLH × wind speed)

UV photolysis proxy

Aerosol attenuation effects

Inspect inference_engine/ for model loading, evaluation, and inference workflows.

Citation / credit

If you reference this work, please cite:

Smart India Hackathon 2025

Data sources: CPCB (India), ERA5/ECMWF, MERRA2/NASA, INSAT

Contact

For questions or corrections, please open a GitHub issue
or contact the repository owner via email: lalithire110@gmail.com
