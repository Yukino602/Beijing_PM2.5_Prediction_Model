# Beijing PM2.5 Next-Hour Forecasting

**Team:** [FILL IN — Phoenix / Brandon Tan Tze Siong, Leng Jaek-Hxiang]
**Competition:** Beijing PM2.5 Forecasting Challenge
**Submission date:** 6 September 2026

## Overview

This repository contains our full pipeline for forecasting next-hour PM2.5 concentration
across 12 Beijing air-quality monitoring stations, using contemporaneous pollutant readings
(PM10, SO2, NO2, CO, O3) and meteorological data (temperature, pressure, dew point, rain,
wind speed/direction) — **without** access to the same-hour PM2.5 reading itself, which is
withheld by design in both the training and test data.

Our final model is a single global LightGBM regressor trained across all stations jointly
(with station identity as a categorical feature), selected after comparing linear models
(Linear/Ridge/Lasso/ElasticNet), RandomForest, and per-station vs. global LightGBM variants.

## Repository Structure

```
notebooks/
  01_data_cleaning_train.ipynb        # Missing-value diagnosis + imputation, training data
  02_data_cleaning_test.ipynb         # Same pipeline applied to test data, id-preserving
  03_training_and_prediction.ipynb    # Feature engineering, model training, validation,
                                       # test inference, submission file generation
submissions/
  submission_v1.csv                   # Exact file corresponding to our leaderboard submission
```

## Methodology

### 1. Data Cleaning & Preprocessing

Both training and test data (12 stations each) contained missing values scattered across
pollutant and weather columns. Our approach:

- **Gap classification**: for each station/variable, missing-value runs were classified by
  duration (`small` < 2h, `medium` < 8h, `long` < 24h).
- **Method comparison**: we benchmarked linear interpolation, PCHIP interpolation, and an
  LGBM-based regression imputer against each other at varying synthetic gap lengths, per
  station and variable, to empirically determine which method performs best at which gap
  length (rather than assuming one method fits all cases).
- **Applied imputation**: small gaps were filled via linear interpolation; gaps beyond each
  variable's empirically-determined crossover point were filled using a trained LightGBM
  regressor (features drawn from other concurrent readings for that station).
- **Test data**: the identical cleaning pipeline was applied to the test set independently,
  with the `id` column preserved via a `(station, observation_timestamp)` join back to the
  raw file, since `id` is not part of the imputation feature set.

### 2. Feature Engineering

All features are derived only from information available at prediction time (no leakage
from the target or from same-hour PM2.5):

- **Lag features**: 1, 2, 3, 6, 12, 24-hour lags for every pollutant and weather variable.
- **Rolling statistics**: mean/std/min/max over 3, 6, 12, 24-hour windows, computed on
  `shift(1)` series to avoid leaking the current hour into its own rolling window.
- **Momentum features**: 1-hour and 3-hour differences, plus 1-hour percent change
  (excluded for RAIN/WSPM/TEMP/DEWP, which can be zero or negative, making percent change
  undefined or physically meaningless).
- **Cyclical time encodings**: sine/cosine transforms of hour-of-day, day-of-week, month,
  and day-of-year, plus raw calendar features and an `is_weekend` flag.
- **Wind direction**: converted from 16-point compass categories to degrees, then to
  sine/cosine components.
- **Interaction features**:
  - `NO2_O3_ratio` — NO2 and O3 have an inverse atmospheric relationship (NO2 depletes
    ozone via a photochemical reaction); this ratio makes that relationship directly
    available to the model.
  - `temp_dewp_diff` and `low_wind_flag` — proxies for atmospheric stagnation, a known
    driver of pollution accumulation.
  - `PM10_lag1_x_WSPM` — captures that wind's dispersal effect depends on how much
    particulate matter was already present.

A critical finding during development: `current_PM2_5` (same-hour PM2.5) was initially
included as a feature and produced an inflated ~0.94 R² through simple autocorrelation.
Once identified as a form of target leakage — the competition's test data does not provide
this value — it was removed from the feature set entirely, and all reported results below
reflect the leak-free version of the pipeline.

### 3. Validation Strategy

`TimeSeriesSplit` (5 folds) was used throughout, ensuring each fold's test period is
strictly chronologically after its training period — appropriate for a forecasting task
where random shuffling would leak future information into training.

### 4. Models Tested

| Model | Approach | Notes |
|---|---|---|
| Linear Regression / Ridge / Lasso / ElasticNet | Per-station, scaled features | Baseline; Lasso/ElasticNet handled collinearity from lag/rolling features better than plain OLS |
| RandomForest | Per-station | Tested as a nonlinear baseline |
| LightGBM (per-station) | 12 separate models | Each station trained on ~4,000–8,000 rows |
| **LightGBM (global) — FINAL** | Single model, all stations combined, `station` as categorical feature | ~360,000 rows total; chosen for materially lower RMSE, since shared atmospheric/chemical relationships across stations are learned from far more data than any single-station model has access to |

### 5. Final Model

**Model:** Single global LightGBM regressor (`lightgbm.LGBMRegressor`)

**Hyperparameters:**
```python
LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=12,
    num_leaves=90,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1
)
```
`station` passed as a native categorical feature (`categorical_feature=['station']`).

No ensembling was used in the final submission.

### 6. Post-Processing

- Predictions clipped at a lower bound of 0 (PM2.5 concentration cannot be physically
  negative; the model can occasionally output small negative values).
- For test-time inference, each station's test period was prefixed with the tail of its
  training history so that lag/rolling-window features have sufficient look-back at the
  start of the test period.

### 7. Results

| Fold | R² | RMSE |
|---|---|---|
| 0 | 0.8476 | 35.736 |
| 1 | 0.8993 | 22.296 |
| 2 | 0.8877 | 24.512 |
| 3 | 0.9134 | 27.268 |
| 4 | 0.8540 | 23.650 |
| **Mean** | **0.8804** | **26.692** |

**Leaderboard score (public, 30% split):** 25.839

Local CV mean RMSE (26.69) and leaderboard RMSE (25.84) are closely aligned, indicating
the time-series validation strategy generalizes well and the model is not overfit to the
validation folds.

### 8. Limitations

- The removal of same-hour PM2.5 (`current_PM2_5`) meaningfully increases task difficulty
  relative to a naive persistence-style approach; our reported metrics reflect this
  harder, leak-free formulation rather than an inflated same-hour-autocorrelation result.
- A single global model was chosen over 12 per-station models for data efficiency; this
  may under-fit station-specific dynamics for stations with unusual local emission sources
  (e.g. industrial vs. residential areas) relative to a fully-tuned per-station approach.
- Hyperparameters were tuned via a small number of manual configurations rather than an
  exhaustive search, due to time constraints; further gains are plausible with formal
  hyperparameter optimization (e.g. Optuna) given more time.
- Test-set imputation quality (interpolation/LGBM-based) is a modeling choice, not ground
  truth — any systematic imputation error propagates into the lag/rolling features used
  for prediction.
- During test inference, a small number of rows per station (37–237 out of ~4,200–4,300,
  i.e. under 6% per station) had residual NaN values in engineered lag/rolling features
  after the train-history lookback join, and were filled with 0 rather than dropped, to
  ensure every required `id` in the submission file received a prediction. Shunyi (237)
  and Huairou (124) had the highest counts, most other stations had 37–94.

## Reproduction Instructions

**Required files:**
- Raw competition data: `train.csv`, `test.csv` (not included in this repo; obtain from
  the competition data source)

**Execution order:**
1. `01_data_cleaning_train.ipynb` — produces `imputed_data/{station}.csv` for each of the
   12 stations from the raw training data.
2. `02_data_cleaning_test.ipynb` — produces `test_cleaned.csv` from the raw test data,
   using the identical gap-classification and imputation methodology.
3. `03_training_and_prediction.ipynb` — loads the cleaned data from steps 1–2, builds
   engineered features, trains the global LightGBM model with 5-fold time-series
   validation, then generates predictions on `test_cleaned.csv` and writes the final
   `submission_v1.csv`.

**Key dependencies:**
```
pandas
numpy
scikit-learn
lightgbm
matplotlib
seaborn
joblib
```
(see `requirements.txt`)

**Random seed:** `random_state=42` used consistently across all model instantiations for
reproducibility.

**Script generating the final submission file:** `03_training_and_prediction.ipynb`
(final cell, writes to `submission_v1.csv`).

## Disclosure

- **External datasets:** None beyond the competition-provided training and test files.
- **External code / repositories consulted:** None directly copied; standard library
  usage (pandas, scikit-learn, LightGBM) per their public documentation.
- **Pretrained models:** None. All models (including the LGBM-based imputation model used
  during data cleaning) were trained from scratch on competition data.
- **AI tools used:** Claude (Anthropic) was used as a coding assistant throughout
  development — for debugging pipeline errors (e.g. duplicate-column bugs, feature
  leakage detection, NaN/inf handling in engineered features), for iterating on model
  architecture (per-station vs. global model), and for structuring this documentation.
  All modeling decisions, hyperparameter choices, and validation design were reviewed and
  directed by the team; code was run and its outputs inspected by the team at each step.
- **Manual modification of predictions:** None. `submission_v1.csv` is the direct,
  unmodified output of the final inference script (aside from the documented clip-at-zero
  post-processing step, which is part of the pipeline itself).
- **Additional information beyond competition-provided files:** None.