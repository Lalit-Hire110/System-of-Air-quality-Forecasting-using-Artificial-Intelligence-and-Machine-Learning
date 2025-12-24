# SAFAL - System of Air Quality Forecasting using AI and ML
## Technical Documentation & Repository Analysis

### 1. Project Overview

**Goal:**  
The primary goal of SAFAL (System of Air Quality Forecasting using AI and ML) is to develop a robust, physics-aware machine learning system for forecasting air quality indices (specifically NO2 and Ozone) in Delhi. The system integrates multi-source data—ground-based sensors (CPCB), meteorological reanalysis (ERA5), and satellite aerosol products (MERRA2)—to overcome the limitations of purely statistical or ground-only approaches.

**Context:**  
This project was developed in the context of the **Smart India Hackathon (SIH) 2025**. It represents a high-intensity, competitive research and development effort by a single contributor focused on the **Data Science and Machine Learning** track.

**Nature of Work:**  
This is a **validated ML + Data Pipeline project**, not a production-ready full-stack application. While a prototype interface (`SIH 2025 Model`) exists for demonstration, the core value lies in the rigorous data engineering, physics-based feature extraction, and model validation pipelines. The repository contains the complete history from raw data ingestion to final model artifacts.

---

### 2. Folder-by-Folder Breakdown

This section categorizes the repository structure to guide archival and cleanup.

#### 📂 `SAFAL_Raw_Data` (CRITICAL - KEEP)
*   **Purpose:** The central hub for data engineering, feature selection, and final dataset generation. This is the "brain" of the data pipeline.
*   **Time Period:** Submission-time to Post-submission (Refined).
*   **Contents:**
    *   `final_engineered_dataset.parquet`: The gold-standard, cleaned, and feature-rich dataset used for training.
    *   `ReadME.txt` & `PLEASE READ ME...`: **Extremely high-value documentation** detailing the physics-based feature engineering (Ventilation Coefficients, UV Photolysis) and data authenticity validation.
    *   `Models and Artifacts`: Trained model binaries and intermediate outputs.
*   **Status:** **Submission-Critical / Core Research**.
*   **Action:** **MUST RETAIN**. This is the most valuable folder for reproducibility.

#### 📂 `SIH 2025 Model` (CRITICAL - KEEP)
*   **Purpose:** The official submission folder containing the executable prototype and inference logic.
*   **Time Period:** Submission-time.
*   **Contents:**
    *   `sih_model_2025.py`: The Streamlit-based application for live scoring and demo.
    *   `models/`, `features/`, `evaluation/`: Production-ready artifacts loaded by the app.
*   **Status:** **Submission-Critical**.
*   **Action:** **MUST RETAIN**. This represents the "product" side of the submission.

#### 📂 `AEP_unified` (SOURCE - ARCHIVE)
*   **Purpose:** The raw data warehouse. Contains the massive unrefined datasets downloaded from various agencies.
*   **Time Period:** Early / Data Ingestion phase.
*   **Contents:**
    *   `Final_CPCB_RAW.csv`, `Final_ERA5_RAW.csv`, `Final_MERRA2_RAW.csv`: The source files before merging.
    *   `pmhw/`: Likely contains high-volume raw data dumps.
*   **Status:** **Data Source**.
*   **Action:** **Retain for Reproducibility (but exclude from git if >100MB)**. These files are necessary to rebuild the dataset from scratch but are likely too large for standard version control.

#### 📂 `APE_07` (LEGACY - DELETE/ARCHIVE)
*   **Purpose:** Early experimental pipeline and legacy code.
*   **Time Period:** Early development (Pre-submission).
*   **Contents:** `week1/`, `robust_aep_pipeline_final.py`, and older documentation.
*   **Status:** **Legacy / Exploratory**.
*   **Action:** **Safe to Delete** (after confirming no unique scripts are needed) or move to a `legacy/` folder.

#### 📂 `cleaned7` (INTERMEDIATE - DELETE)
*   **Purpose:** Intermediate station-level data cleaning and merging.
*   **Time Period:** Mid-development.
*   **Contents:** Scripts like `merging_stations_delhi.py`.
*   **Status:** **Intermediate**.
*   **Action:** **Safe to Delete** (assuming logic is superseded by `SAFAL_Raw_Data` pipelines).

#### 📂 `station_coordinates_csv` (METADATA - KEEP)
*   **Purpose:** Geographic metadata for stations.
*   **Status:** **Helper Data**.
*   **Action:** **Retain**. Useful for visualization and spatial analysis.

---

### 3. Data Lineage & Pipelines

The project implements a sophisticated ETL (Extract, Transform, Load) pipeline that merges disparate data sources into a unified analytical dataset.

**Data Sources:**
1.  **CPCB (Central Pollution Control Board):** Ground-truth air quality measurements (PM2.5, PM10, NO2, Ozone) from 5 key stations in Delhi.
2.  **ERA5 (ECMWF Reanalysis v5):** High-resolution meteorological data (Temperature, Wind Speed, Boundary Layer Height).
3.  **MERRA2 (NASA):** Satellite-based aerosol products (Aerosol Optical Depth, Black Carbon, Dust).

**The Pipeline Flow:**
1.  **Ingestion:** Raw data is collected in `AEP_unified`.
2.  **Cleaning & Merging:** Timestamps are aligned (handling UTC vs IST offsets), and missing values are imputed using forward-fill for meteorological data (validated in `SAFAL_Raw_Data/ReadME.txt`).
3.  **Feature Engineering (`SAFAL_Raw_Data`):**
    *   **Physics-Based Features:**
        *   *Ventilation Coefficient (VC)*: `BLH * Wind Speed` (measures dispersion capacity).
        *   *UV Photolysis Proxy*: `Solar Radiation * cos(Solar Zenith Angle)` (drivers of Ozone formation).
        *   *Aerosol Attenuation*: Adjusting UV availability based on AOD (Aerosol Optical Depth).
4.  **Final Dataset:** The processed data is saved as `final_engineered_dataset.parquet` in `SAFAL_Raw_Data`.

**Note on Data Volume:**
Raw satellite and reanalysis files (in `AEP_unified`) are voluminous. The final parquet file is highly compressed and optimized for ML training.

---

### 4. Modeling Summary

**Models Developed:**
*   **Primary Targets:** NO2 (Nitrogen Dioxide) and O3 (Ozone).
*   **Algorithm:** Gradient Boosting (XGBoost/LightGBM) and Quantile Regression for uncertainty estimation.
*   **Location:** `SIH 2025 Model/models/`.

**Key Characteristics:**
*   **Scope:** The models cover a 6-year period (2019-2024), with a chronological split (Train: 2019-2022, Val: 2023, Test: 2024) to respect temporal dependencies.
*   **Validation:** Extensive validation was performed to ensure physical consistency (e.g., ensuring Ozone predictions correlate correctly with Temperature and UV).
*   **Artifacts:** The system produces not just point predictions but also **Confidence Intervals (q10-q90)**, essential for risk assessment.

---

### 5. Ownership & Scope Clarification

**Contributor Role:**
The contributor acted as the **Lead Data Scientist and ML Engineer**.
*   **Owned:** Data acquisition, cleaning strategy, physics-based feature engineering, model architecture, training, and validation.
*   **Scope Boundary:** The project focuses on the **scientific validity and algorithmic performance** of the forecasting system. Full-stack deployment (cloud hosting, CI/CD, user management) was explicitly outside the scope of this research-centric submission.

---

### 6. How This Repository Should Be Interpreted

**For Reviewers & Recruiters:**

1.  **Focus on `SAFAL_Raw_Data`:** This folder demonstrates the depth of domain knowledge. The `ReadME.txt` files here are not generic; they prove a deep understanding of atmospheric physics and data integrity.
2.  **Reproducibility:** Unlike many hackathon projects that are "black boxes", this repository contains the raw ingredients and the recipe (code) to reproduce the scientific results.
3.  **Scale:** The project handles high-frequency (hourly) data over 6 years across multiple modalities (Ground + Satellite + Reanalysis), showcasing ability to handle complex, real-world datasets.
4.  **Validation Rigor:** The analysis goes beyond simple accuracy metrics (RMSE) to verify *physical plausibility* (e.g., "Does the model know that rain cleans the air?"), which is a hallmark of mature data science work.

**Recommendation for GitHub:**
*   Root the repo at a clean level.
*   Keep `SAFAL_Raw_Data` (as `data_pipeline`) and `SIH 2025 Model` (as `app`).
*   Add this document as `TECHNICAL_REPORT.md`.
*   Use `.gitignore` to exclude the massive CSVs in `AEP_unified` while keeping the scripts.
