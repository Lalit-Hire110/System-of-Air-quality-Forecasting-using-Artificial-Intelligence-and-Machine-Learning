# Smart India Hackathon 2025 - Air Quality Prediction Models

Generated on: 2025-10-08 18:05:16

## Project Overview

This repository contains trained machine learning models for predicting ground-level NO2 and O3 concentrations using multi-source environmental data including:
- CPCB ground station measurements (canonical targets)
- ERA5 reanalysis meteorological data
- MERRA-2 atmospheric data  
- Satellite observations
- Traffic and emission proxies

## Model Performance

### NO2 Model
- **Validation**: MAE=2.68 ug/m3, RMSE=7.33 ug/m3, R2=0.916
- **Holdout**: MAE=4.42 ug/m3, RMSE=9.56 ug/m3, R2=0.850
- **Features**: 20 selected features
- **Samples**: 3,458 holdout predictions

### O3 Model  
- **Validation**: MAE=11.02 ug/m3, RMSE=18.62 ug/m3, R2=0.566
- **Holdout**: MAE=7.93 ug/m3, RMSE=12.01 ug/m3, R2=0.692
- **Features**: 20 selected features
- **Samples**: 3,203 holdout predictions

## Key Design Decisions

### Target Selection
- **CPCB-First Policy**: Used CPCB ground measurements as canonical targets
- **SIH Exclusion**: Excluded SIH-provided train_* targets from supervised learning
- **Target Columns**: NO2 (ug/m3), Ozone (ug/m3)

### Temporal Validation
- **Holdout Strategy**: Final 30 days per station (time-aware split)
- **Cross-Validation**: Time-ordered blocks with 6-24h purge windows
- **Leakage Prevention**: Strict temporal ordering, no look-ahead features

### Feature Engineering
- **Multi-Source Integration**: ERA5, MERRA-2, satellite, ground sensors
- **Lag Features**: 1-24h temporal lags for meteorological variables
- **Rolling Statistics**: 3h, 6h, 12h, 24h rolling aggregates
- **Missing Data**: Dropped >70% missing, careful imputation 40-70%, standard <40%

## Directory Structure

/SIH 2025 Model/
├── models/ # Trained model artifacts
│   ├── no2_model.pkl # NO2 LightGBM model
│   ├── no2_imputer.pkl # NO2 feature imputer
│   ├── o3_model.pkl # O3 LightGBM model
│   ├── o3_imputer.pkl # O3 feature imputer
│   ├── no2_quantile_models.pkl # NO2 uncertainty models (if available)
│   └── o3_quantile_models.pkl # O3 uncertainty models (if available)
├── features/ # Feature selection metadata
│   ├── feature_metadata.json # Selection process summary
│   ├── no2_features.csv # Final NO2 feature list
│   └── o3_features.csv # Final O3 feature list
├── predictions/ # Model predictions
│   ├── no2_holdout_predictions.csv # NO2 holdout results with uncertainties
│   └── o3_holdout_predictions.csv # O3 holdout results with uncertainties
├── evaluation/ # Performance metrics
│   ├── evaluation_summary.json # Overall model performance
│   ├── no2_station_metrics.csv # Per-station NO2 performance
│   ├── o3_station_metrics.csv # Per-station O3 performance
│   └── dataset_splits.csv # Train/holdout split information
└── documentation/ # This README and additional docs
    └── README.txt

## Model Usage

### Loading Models
import pickle
import pandas as pd

Load NO2 model
with open('models/no2_model.pkl', 'rb') as f:
no2_model = pickle.load(f)
with open('models/no2_imputer.pkl', 'rb') as f:
no2_imputer = pickle.load(f)

Load feature list
no2_features = pd.read_csv('features/no2_features.csv')['feature'].tolist()

### Making Predictions  
Prepare input data with exact feature names
X = input_data[no2_features]
X_imputed = pd.DataFrame(
no2_imputer.transform(X),
columns=no2_features,
index=X.index
)
predictions = no2_model.predict(X_imputed)

## Data Dependencies

### Required Input Features
- **Meteorological**: Temperature, humidity, wind speed/direction, pressure
- **Atmospheric**: Boundary layer height, solar radiation, precipitation  
- **Chemical**: Co-pollutant concentrations (NOx, SO2, PM10, NH3)
- **Spatial**: Latitude, longitude, station characteristics
- **Temporal**: Lag and rolling window features

### Feature Engineering Notes
- Lag features require 1-24h historical data
- Rolling statistics need 3-24h windows
- Missing values handled via median imputation
- Categorical features encoded during preprocessing

## Validation Results

### Leakage Testing
- **Large Purge CV**: Stable performance with 48-72h purge windows
- **Covariate Shift**: +1h feature shift reduced accuracy as expected  
- **Co-pollutant Ablation**: Performance depends on contemporaneous measurements

### Station-Level Performance
- **NO2**: Best at stations with consistent measurements
- **O3**: Best at stations with stable meteorological conditions
- **Heterogeneity**: Consider station-specific calibration for deployment

## Deployment Considerations

### Nowcasting vs Forecasting
- **Current Setup**: Optimized for nowcasting with contemporaneous inputs
- **Co-pollutant Dependency**: NO2 performance relies on same-hour measurements
- **Forecasting Mode**: Remove contemporaneous co-pollutants, expect lower accuracy

### Data Quality Requirements
- **Completeness**: <40% missing values preferred for reliable predictions
- **Temporal Alignment**: Ensure proper timestamp handling across data sources
- **Feature Consistency**: Maintain exact feature names and preprocessing

## Technical Specifications

- **Framework**: LightGBM with early stopping
- **Hyperparameters**: 4000 estimators, 0.02 learning rate, 96 leaves
- **Validation**: Time-ordered cross-validation with temporal purge
- **Uncertainty**: Quantile regression (10th/90th percentiles) where available
- **Environment**: Python 3.7+, scikit-learn, pandas, numpy, lightgbm

## Citation and Contact

Developed for Smart India Hackathon 2025 Air Quality Challenge.
Framework follows CPCB-first target policy with time-aware validation.

For questions about model usage or performance, refer to evaluation metrics
and per-station results in the evaluation/ directory.

---
Generated by SIH Air Quality Pipeline v1.0
Last updated: 2025-10-08 18:05:16
